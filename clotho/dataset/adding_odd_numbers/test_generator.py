import random
import json

def generate_integer_sequences(max_number=1000, length=10, num_sequences=5, allow_duplicates=False):
    """
    Generate a list of integer sequences with specified properties.

    Args:
        max_number (int): The maximum number in the sequence.
        length (int): The length of each sequence.
        num_sequences (int): The number of sequences to generate.
        allow_duplicates (bool): Whether to allow duplicate numbers in the sequences.

    Returns:
        list: A list of integer sequences.
    """

    sequences = []
    for _ in range(num_sequences):
        if allow_duplicates:
            sequence = [random.randint(1, max_number) for _ in range(length)]
        else:
            sequence = random.sample(range(1, max_number + 1), length)
        sequences.append(sequence)

    return sequences

def sum_of_odd_numbers(sequence):
    return sum(num for num in sequence if num % 2 != 0)

def generate_balanced_test_suite():
    lengths = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    max_numbers = [100, 1000, 10000]
    
    dataset = []
    for length in lengths:
        for max_number in max_numbers:
            for allow_duplicates in [True, False]:
                num_sequences = 100
                print(f'length = {length}, max_number = {max_number}, allow_duplicates = {allow_duplicates}, num_sequences = {num_sequences}')
                sequences = generate_integer_sequences(max_number=max_number, length=length, num_sequences=num_sequences, allow_duplicates=allow_duplicates)
                for seq in sequences:
                    dataset.append({
                        "sequence": seq,
                        "sum_of_odds": sum_of_odd_numbers(seq),
                        "length": length,
                        "max_number": max_number,
                        "allow_duplicates": allow_duplicates
                    })
    
    print(f"Generated {len(dataset)} test cases.")
    return dataset
    

if __name__ == "__main__":
    # Example usage
    dataset = generate_balanced_test_suite()
    with open('integer_sequences_length_1_to_10_uniform.jsonl', 'w') as f:
        for item in dataset:
            f.write(f"{json.dumps(item)}\n")
    