import os

def display_tree_structure_and_file_count(start_path, depth=0):
    total_files = 0

    # Iterate over items in the directory
    for item in os.listdir(start_path):
        item_path = os.path.join(start_path, item)

        # If the item is a directory, recursively traverse its contents and accumulate the file count
        if os.path.isdir(item_path):
            #print("  " * depth + "├── " + item)
            sub_dir_files, sub_dir_total = display_tree_structure_and_file_count(item_path, depth + 1)
            total_files += sub_dir_files
            print("  " * (depth + 1) + f"└── Total files in {item}: {sub_dir_files}")
        else:
            total_files += 1
            #print("  " * depth + "├── " + item)

    return total_files, depth

start_path = '성준원'  # Replace this with the path of the directory you want to traverse
total_files, _ = display_tree_structure_and_file_count(start_path)
print(f"Total files in {start_path}: {total_files}")
