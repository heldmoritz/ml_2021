import json


def convert2json(filename: str, n_recs: list):
    """Function to convert the .test and .train files to JSON.
    :param filename: path to file containing raw data
    :param n_recs: number of recordings per speaker
    """

    # Read the raw file
    with open(filename, "r", encoding="utf-8") as f:
        data = {}
        speaker_idx = 0
        recording_idx = 0
        for line in f:
            # Skip empty lines in ae.test
            if not line.strip():
                continue

            # Convert string to list of floats
            line_data = list(map(float, line.strip().split(" ")))

            # Add data to current recording if not all values equal 1.0
            if not all([x == 1.0 for x in line_data]):
                if f"speaker{speaker_idx}" not in data:
                    data[f"speaker{speaker_idx}"] = {f"recording{recording_idx}": [line_data]}
                elif f"recording{recording_idx}" not in data[f"speaker{speaker_idx}"]:
                    data[f"speaker{speaker_idx}"][f"recording{recording_idx}"] = [line_data]
                else:
                    data[f"speaker{speaker_idx}"][f"recording{recording_idx}"].append(line_data)
            # End of recording reached, increment recording index or move to next speaker index
            else:
                recording_idx += 1
                # If the number of recordings for this speaker has been reached,
                # increment speaker index and reset recording index
                if recording_idx == n_recs[speaker_idx]:
                    speaker_idx += 1
                    recording_idx = 0

    # Write the JSON file
    with open(filename.replace(".", "_") + ".json", "w", encoding="utf-8") as f:
        json.dump(data, f)


if __name__ == '__main__':
    convert2json("ae.train", [30]*9)
    convert2json("ae.test", [31, 35, 88, 44, 29, 24, 40, 50, 29])
