/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package controller

import (
	"context"
	"fmt"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/tools/record"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/builder"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/event"
	"sigs.k8s.io/controller-runtime/pkg/log"
	"sigs.k8s.io/controller-runtime/pkg/predicate"

	nvidiacomv1alpha1 "github.com/ai-dynamo/dynamo/deploy/cloud/operator/api/v1alpha1"
	commonController "github.com/ai-dynamo/dynamo/deploy/cloud/operator/internal/controller_common"
	"github.com/ai-dynamo/dynamo/deploy/cloud/operator/internal/dynamo"
)

const (
	FailedState  = "failed"
	ReadyState   = "successful"
	PendingState = "pending"
)

type etcdStorage interface {
	DeleteKeys(ctx context.Context, prefix string) error
}

// DynamoGraphDeploymentReconciler reconciles a DynamoGraphDeployment object
type DynamoGraphDeploymentReconciler struct {
	client.Client
	Config                     commonController.Config
	Recorder                   record.EventRecorder
	VirtualServiceGateway      string
	IngressControllerClassName string
	IngressControllerTLSSecret string
	IngressHostSuffix          string
}

// +kubebuilder:rbac:groups=nvidia.com,resources=dynamographdeployments,verbs=get;list;watch;create;update;patch;delete
// +kubebuilder:rbac:groups=nvidia.com,resources=dynamographdeployments/status,verbs=get;update;patch
// +kubebuilder:rbac:groups=nvidia.com,resources=dynamographdeployments/finalizers,verbs=update

// Reconcile is part of the main kubernetes reconciliation loop which aims to
// move the current state of the cluster closer to the desired state.
// TODO(user): Modify the Reconcile function to compare the state specified by
// the DynamoGraphDeployment object against the actual cluster state, and then
// perform operations to make the cluster state reflect the state specified by
// the user.
//
// For more details, check Reconcile and its Result here:
// - https://pkg.go.dev/sigs.k8s.io/controller-runtime@v0.19.1/pkg/reconcile
func (r *DynamoGraphDeploymentReconciler) Reconcile(ctx context.Context, req ctrl.Request) (ctrl.Result, error) {
	logger := log.FromContext(ctx)

	var err error
	reason := "undefined"
	message := ""
	readyStatus := metav1.ConditionFalse
	// retrieve the CRD
	dynamoDeployment := &nvidiacomv1alpha1.DynamoGraphDeployment{}
	if err = r.Get(ctx, req.NamespacedName, dynamoDeployment); err != nil {
		return ctrl.Result{}, client.IgnoreNotFound(err)
	}
	if err != nil {
		// not found, nothing to do
		return ctrl.Result{}, nil
	}

	defer func() {
		if err != nil {
			dynamoDeployment.SetState(FailedState)
			message = err.Error()
			logger.Error(err, "Reconciliation failed")
		}
		// update the CRD status condition
		dynamoDeployment.AddStatusCondition(metav1.Condition{
			Type:               "Ready",
			Status:             readyStatus,
			Reason:             reason,
			Message:            message,
			LastTransitionTime: metav1.Now(),
		})
		err = r.Status().Update(ctx, dynamoDeployment)
		if err != nil {
			logger.Error(err, "Unable to update the CRD status", "crd", req.NamespacedName)
		}
		logger.Info("Reconciliation done")
	}()

	deleted, err := commonController.HandleFinalizer(ctx, dynamoDeployment, r.Client, r)
	if err != nil {
		logger.Error(err, "failed to handle the finalizer")
		reason = "failed_to_handle_the_finalizer"
		return ctrl.Result{}, err
	}
	if deleted {
		return ctrl.Result{}, nil
	}

	// fetch the dynamoGraphConfig
	dynamoGraphConfig, err := dynamo.GetDynamoGraphConfig(ctx, dynamoDeployment, r.Recorder)
	if err != nil {
		logger.Error(err, "failed to get the DynamoGraphConfig")
		reason = "failed_to_get_the_DynamoGraphConfig"
		return ctrl.Result{}, err
	}

	// generate the dynamoComponentsDeployments from the config
	dynamoComponentsDeployments, err := dynamo.GenerateDynamoComponentsDeployments(ctx, dynamoDeployment, dynamoGraphConfig, r.generateDefaultIngressSpec(dynamoDeployment))
	if err != nil {
		logger.Error(err, "failed to generate the DynamoComponentsDeployments")
		reason = "failed_to_generate_the_DynamoComponentsDeployments"
		return ctrl.Result{}, err
	}

	// merge the dynamoComponentsDeployments with the dynamoComponentsDeployments from the CRD
	for _, deployment := range dynamoComponentsDeployments {
		if deployment.Spec.Ingress.Enabled {
			dynamoDeployment.SetEndpointStatus(r.isEndpointSecured(), getIngressHost(deployment.Spec.Ingress))
		}
	}

	// reconcile the dynamoComponent
	// for now we use the same component for all the services and we differentiate them by the service name when launching the component
	dynamoComponent := &nvidiacomv1alpha1.DynamoComponent{
		ObjectMeta: metav1.ObjectMeta{
			Name:      getK8sName(dynamoDeployment.Spec.DynamoGraph),
			Namespace: dynamoDeployment.Namespace,
		},
		Spec: nvidiacomv1alpha1.DynamoComponentSpec{
			DynamoComponent: dynamoDeployment.Spec.DynamoGraph,
		},
	}
	_, dynamoComponent, err = commonController.SyncResource(ctx, r, dynamoDeployment, func(ctx context.Context) (*nvidiacomv1alpha1.DynamoComponent, bool, error) {
		return dynamoComponent, false, nil
	})
	if err != nil {
		logger.Error(err, "failed to sync the DynamoComponent")
		reason = "failed_to_sync_the_DynamoComponent"
		return ctrl.Result{}, err
	}
	if !dynamoComponent.IsReady() {
		logger.Info("The DynamoComponent is not ready")
		reason = "dynamoComponent_is_not_ready"
		message = "The DynamoComponent is not ready"
		readyStatus = metav1.ConditionFalse
		return ctrl.Result{}, nil
	}

	notReadyDeployments := []string{}
	// reconcile the dynamoComponentsDeployments
	for serviceName, dynamoComponentDeployment := range dynamoComponentsDeployments {
		logger.Info("Reconciling the DynamoComponentDeployment", "serviceName", serviceName, "dynamoComponentDeployment", dynamoComponentDeployment)
		_, dynamoComponentDeployment, err = commonController.SyncResource(ctx, r, dynamoDeployment, func(ctx context.Context) (*nvidiacomv1alpha1.DynamoComponentDeployment, bool, error) {
			return dynamoComponentDeployment, false, nil
		})
		if err != nil {
			logger.Error(err, "failed to sync the DynamoComponentDeployment")
			reason = "failed_to_sync_the_DynamoComponentDeployment"
			return ctrl.Result{}, err
		}
		if !dynamoComponentDeployment.Status.IsReady() {
			notReadyDeployments = append(notReadyDeployments, dynamoComponentDeployment.Name)
		}
	}
	if len(notReadyDeployments) == 0 {
		dynamoDeployment.SetState(ReadyState)
		reason = "all_deployments_are_ready"
		message = "All deployments are ready"
		readyStatus = metav1.ConditionTrue
	} else {
		reason = "some_deployments_are_not_ready"
		message = fmt.Sprintf("The following deployments are not ready: %v", notReadyDeployments)
		dynamoDeployment.SetState(PendingState)
	}

	return ctrl.Result{}, nil

}

func (r *DynamoGraphDeploymentReconciler) generateDefaultIngressSpec(dynamoDeployment *nvidiacomv1alpha1.DynamoGraphDeployment) *nvidiacomv1alpha1.IngressSpec {
	res := &nvidiacomv1alpha1.IngressSpec{
		Enabled:           r.VirtualServiceGateway != "" || r.IngressControllerClassName != "",
		Host:              dynamoDeployment.Name,
		UseVirtualService: r.VirtualServiceGateway != "",
	}
	if r.IngressControllerClassName != "" {
		res.IngressControllerClassName = &r.IngressControllerClassName
	}
	if r.IngressControllerTLSSecret != "" {
		res.TLS = &nvidiacomv1alpha1.IngressTLSSpec{
			SecretName: r.IngressControllerTLSSecret,
		}
	}
	if r.IngressHostSuffix != "" {
		res.HostSuffix = &r.IngressHostSuffix
	}
	if r.VirtualServiceGateway != "" {
		res.VirtualServiceGateway = &r.VirtualServiceGateway
	}
	return res
}

func (r *DynamoGraphDeploymentReconciler) isEndpointSecured() bool {
	if r.VirtualServiceGateway != "" && r.Config.VirtualServiceSupportsHTTPS {
		return true
	}
	return r.IngressControllerTLSSecret != ""
}

func (r *DynamoGraphDeploymentReconciler) FinalizeResource(ctx context.Context, dynamoDeployment *nvidiacomv1alpha1.DynamoGraphDeployment) error {
	// for now doing nothing
	return nil
}

// SetupWithManager sets up the controller with the Manager.
func (r *DynamoGraphDeploymentReconciler) SetupWithManager(mgr ctrl.Manager) error {
	return ctrl.NewControllerManagedBy(mgr).
		For(&nvidiacomv1alpha1.DynamoGraphDeployment{}, builder.WithPredicates(
			predicate.GenerationChangedPredicate{},
		)).
		Named("dynamographdeployment").
		Owns(&nvidiacomv1alpha1.DynamoComponentDeployment{}, builder.WithPredicates(predicate.Funcs{
			// ignore creation cause we don't want to be called again after we create the deployment
			CreateFunc:  func(ce event.CreateEvent) bool { return false },
			DeleteFunc:  func(de event.DeleteEvent) bool { return true },
			UpdateFunc:  func(de event.UpdateEvent) bool { return true },
			GenericFunc: func(ge event.GenericEvent) bool { return true },
		})).
		Owns(&nvidiacomv1alpha1.DynamoComponent{}, builder.WithPredicates(predicate.Funcs{
			// ignore creation cause we don't want to be called again after we create the deployment
			CreateFunc:  func(ce event.CreateEvent) bool { return false },
			DeleteFunc:  func(de event.DeleteEvent) bool { return true },
			UpdateFunc:  func(de event.UpdateEvent) bool { return true },
			GenericFunc: func(ge event.GenericEvent) bool { return true },
		})).
		WithEventFilter(commonController.EphemeralDeploymentEventFilter(r.Config)).
		Complete(r)
}

func (r *DynamoGraphDeploymentReconciler) GetRecorder() record.EventRecorder {
	return r.Recorder
}
