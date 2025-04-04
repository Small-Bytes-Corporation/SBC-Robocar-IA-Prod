# **************************************************************************** #
#                                                                              #
#                        Robocar - Data Buffer (Recording)                     #
#                                                                              #
# **************************************************************************** #

import csv
import json
import os
from datetime import datetime
from settings import NUM_RAYS, DEFAULT_CSV_PATH

# **************************************************************************** #
#                                                                              #
#                            DATA BUFFER CLASS                                 #
#                                                                              #
# **************************************************************************** #

class DataBufferNR:
    def __init__(self):
        self.buffer = []

    def add_to_buffer(self, data_map, values):
        normalized_state = self.normalize_state(data_map)
        self.buffer.append({
            'state': normalized_state,
            'values': values
        })

    def normalize_state(self, data_map):
        lidar = [float(x) / 500.0 for x in data_map['lidar']]
        return lidar[:NUM_RAYS]

    def save_data(self):
        os.makedirs(DEFAULT_CSV_PATH, exist_ok=True)
        filename = f"{DEFAULT_CSV_PATH}{datetime.now().strftime('%Y%m%d_%H%M%S')}_data.csv"
        if not self.buffer:
            print("No data to save.")
            return filename
        fieldnames = ['state', 'values']

        try:
            with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                for record in self.buffer:
                    writer.writerow({
                        'state': json.dumps(record['state']),
                        'values': json.dumps(record['values'])
                    })
            print(f"Data saved to {filename}")
        except Exception as e:
            print(f"Error saving data: {e}")
        finally:
            self.buffer.clear()

        return filename
