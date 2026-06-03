import random


def sample_lines(input_file, output_file, sample_size=500, seed=42):
    """Randomly samples a specific number of lines from a text file with a fixed seed."""
    # Set the seed for reproducibility
    random.seed(seed)

    try:
        # Read all lines from the source file
        with open(input_file, "r", encoding="utf-8") as f:
            lines = f.readlines()

        # Check if the file has enough lines to sample from
        if len(lines) < sample_size:
            print(
                f"Warning: The input file only has {len(lines)} lines. Sampling all of them."
            )
            sample_size = len(lines)

        # Randomly sample lines without replacement
        sampled_lines = random.sample(lines, sample_size)

        # Write the sampled lines to the new file
        with open(output_file, "w", encoding="utf-8") as f:
            f.writelines(sampled_lines)

        print(f"Successfully sampled {sample_size} lines and saved to '{output_file}'.")

    except FileNotFoundError:
        print(f"Error: The file '{input_file}' was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")


# --- Configuration ---
INPUT_FILENAME = "datasets/stackoverflow/title_StackOverflow_full.txt"
OUTPUT_FILENAME = "datasets/stackoverflow/title_StackOverflow.txt"
RANDOM_SEED = 20260601

# Run the function
sample_lines(INPUT_FILENAME, OUTPUT_FILENAME, sample_size=500, seed=RANDOM_SEED)
