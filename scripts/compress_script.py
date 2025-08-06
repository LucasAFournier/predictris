import csv
import json
import random
import sys

def set_max_csv_field_size():
    """
    Increases the CSV field size limit to the maximum possible for the system.
    This is necessary for reading CSVs with extremely large fields.
    """
    max_int = 1000000 # 131072
    while True:
        # Decrease the max_int value by half until it's small enough.
        try:
            csv.field_size_limit(max_int)
            print(f"CSV field size limit set to {max_int}")
            break
        except OverflowError:
            max_int = int(max_int / 2)

def compress_csv_values(sample_size=1000):
    """
    Reads a CSV file, and for rows where the 'value' column contains a list
    of more than `sample_size` items, it randomly samples that many items.
    """
    # Set the CSV field size limit to handle large fields
    set_max_csv_field_size()

    input_path = r'c:\Users\lucas\Desktop\GitHub\predictris\plots\tetrominos=JS_depth=2\metrics\confidences\0.csv'
    output_path = r'c:\Users\lucas\Desktop\GitHub\predictris\plots\tetrominos=JS_depth=2\metrics\confidences\compressed.csv'

    try:
        with open(input_path, mode='r', newline='', encoding='utf-8') as infile, \
             open(output_path, mode='w', newline='', encoding='utf-8') as outfile:

            reader = csv.DictReader(infile)
            fieldnames = reader.fieldnames
            if not fieldnames:
                print("Error: CSV file has no header or is empty.")
                return
                
            writer = csv.DictWriter(outfile, fieldnames=fieldnames)
            writer.writeheader()

            print("Starting file processing...")
            # Process each data row
            for i, row in enumerate(reader):
                try:
                    value_str = row.get('value', '')
                    
                    value_list = json.loads(value_str)

                    if isinstance(value_list, list) and len(value_list) > sample_size:
                        sampled_list = random.sample(value_list, sample_size)
                        row['value'] = json.dumps(sampled_list)
                        # Adding a dot to show progress without flooding the console
                        print('.', end='', flush=True)
                    
                    writer.writerow(row)

                except (json.JSONDecodeError, TypeError):
                    # This will handle rows where 'value' is not a valid JSON list
                    writer.writerow(row)
                except Exception as e:
                    print(f"\nAn error occurred on row {i+2}: {e}. Writing original row.")
                    writer.writerow(row)
        
        print(f"\n\nProcessing complete. Compressed file saved as '{output_path}'")

    except FileNotFoundError:
        print(f"Error: The file at '{input_path}' was not found.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

# Run the function
compress_csv_values()