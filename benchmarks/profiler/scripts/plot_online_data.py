import matplotlib.pyplot as plt
import random
import json
import re

def parse_online_log_file(filename, num_requests, shuffle = False, max_sample_request_tokens: int = 16384):
    with open(filename, 'r', encoding='utf-8') as file:
        data = file.read()

    request_blocks = re.split(r"=== Line \d+ ===", data)
    if shuffle: 
        random.shuffle(request_blocks)

    samples = []
    for i, request_data in enumerate(request_blocks):
        if "Request Body:" in request_data:
            try:
                request_body_str = request_data.split("Request Body:")[1].split("Output Tokens:")[0].strip()
                request_body = json.loads(request_body_str)
                input_prompt = ""
                output_content = request_data.split("Request Body:")[1].split("Output Tokens:")[1].strip()
                if "prompt" in request_body.keys():
                    input_prompt = request_body["prompt"]
                else:
                    input_prompt = ''.join(str(s["content"]) for s in request_body["messages"])
                if len(input_prompt) + len(output_content) > max_sample_request_tokens: continue
                item = (input_prompt, len(input_prompt), len(output_content))
                samples.append(item)
            except (json.JSONDecodeError, IndexError) as e:
                print(f"解析错误：{e}")
        if len(samples) >= num_requests: break
    average_input_len = sum(sample[1] for sample in samples) / len(samples)
    average_output_len = sum(sample[2] for sample in samples) / len(samples)
    print(f"Requests average input_len={average_input_len}, output_len={average_output_len}")
    return samples

def plot_length_distribution(samples, bins=50):
    """
    Plot the distribution of input and output token lengths from samples.
    """
    # Extract lengths
    input_lengths = [s[1] for s in samples]
    output_lengths = [s[2] for s in samples]

    # Create histogram
    plt.figure(figsize=(10, 6))
    plt.hist(input_lengths, bins=bins, alpha=0.6, label='Input Length')
    plt.hist(output_lengths, bins=bins, alpha=0.6, label='Output Length')

    plt.xlabel('Token Length')
    plt.ylabel('Frequency')
    plt.title('Distribution of Input and Output Token Lengths')
    plt.legend()
    plt.tight_layout()
    plt.savefig("online_hist_data.png")
    plt.show()

def plot_length_bar(samples):
    """
    Plot the distribution of input and output token lengths from samples.
    """
    # Extract lengths
    indices = list(range(1, len(samples) + 1))
    input_lengths = [s[1] for s in samples]
    output_lengths = [s[2] for s in samples]

    # Create histogram
    plt.figure(figsize=(10, 6))
    plt.bar(indices, input_lengths, alpha=0.6, label='Input Length')
    plt.bar(indices, output_lengths, alpha=0.6, label='Output Length')

    plt.xlabel('index')
    plt.ylabel('length')
    plt.title('Distribution of Input and Output Token Lengths')
    plt.legend()
    plt.tight_layout()
    plt.savefig("online_bar_data.png")
    plt.show()

def plot_length_sequence(samples):
    """
    Plot the input and output token lengths by sample index in the given sample order.
    """
    # Extract lengths
    input_lengths = [s[1] for s in samples]
    output_lengths = [s[2] for s in samples]

    # Sample indices
    indices = list(range(1, len(samples) + 1))

    plt.figure(figsize=(12, 6))
    plt.plot(indices, input_lengths, marker='o', linestyle='-', label='Input Length')
    plt.plot(indices, output_lengths, marker='x', linestyle='--', label='Output Length')

    plt.xlabel('Sample Index')
    plt.ylabel('Token Length')
    plt.title('Input and Output Token Lengths by Sample Order')
    plt.legend()
    plt.tight_layout()
    plt.savefig("online_plot_data.png")
    plt.show()
def plot_length_waveform(samples):
    """
    Plot input and output token lengths as waveforms (stem plots) by sample index.
    """
    indices = list(range(1, len(samples) + 1))
    input_lengths = [s[1] for s in samples]
    output_lengths = [s[2] for s in samples]

    plt.figure(figsize=(12, 6))
    plt.stem(indices, input_lengths, linefmt='-', markerfmt='o', basefmt=' ', label='Input Length')
    plt.stem(indices, output_lengths, linefmt='--', markerfmt='x', basefmt=' ', label='Output Length')

    plt.xlabel('Sample Index')
    plt.ylabel('Token Length')
    plt.title('Waveform of Input and Output Token Lengths')
    plt.legend()
    plt.tight_layout()
    plt.savefig("online_stem_data.png")
    plt.show()

def main():
    file_path = "/workspace/examples/llm/llama-70b-completions_online_dataset.json"
    samples = parse_online_log_file(file_path, num_requests=2048)
    if samples:
        plot_length_bar(samples)

if __name__ == "__main__":
    main()