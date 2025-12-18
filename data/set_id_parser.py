import csv

# Configuration
# Change this to the theme name you want
# Ensure it exists in the themes.csv file
THEME_NAME = "Speed Champions"
THEMES_FILE = "../files/themes.csv"
SETS_FILE = "../files/sets.csv"

# Set the output file path
OUTPUT_FILE = "../files/" + THEME_NAME + "_set_ids.txt"


def find_theme_id(theme_name, themes_file):
    """Find the theme ID for a given theme name."""
    with open(themes_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['name'] == theme_name:
                return row['id']
    return None


def get_set_ids_by_theme(theme_id, sets_file):
    """Get all set IDs for a given theme ID."""
    set_ids = []
    with open(sets_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['theme_id'] == theme_id:
                set_ids.append(row['set_num'])
    return set_ids


def write_set_ids_to_file(set_ids, output_file):
    """Write set IDs to a text file, one per line."""
    with open(output_file, 'w', encoding='utf-8') as f:
        for set_id in set_ids:
            f.write(set_id + '\n')


def main():
    """Main function to extract set IDs for a theme."""
    # Find theme ID
    theme_id = find_theme_id(THEME_NAME, THEMES_FILE)
    if theme_id is None:
        print(f"Theme '{THEME_NAME}' not found!")
        return
    
    print(f"Found theme '{THEME_NAME}' with ID: {theme_id}")
    
    # Get set IDs
    set_ids = get_set_ids_by_theme(theme_id, SETS_FILE)
    print(f"Found {len(set_ids)} sets")
    
    # Write to file
    write_set_ids_to_file(set_ids, OUTPUT_FILE)
    print(f"Set IDs written to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
