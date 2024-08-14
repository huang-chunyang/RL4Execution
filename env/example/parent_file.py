import subprocess
import json

result = subprocess.run(
    ['python', './child_file.py' ], 
    capture_output=True, text=True
)
output_lines = result.stdout.splitlines()
json_output_lines = []
capture_json = False
for line in output_lines:
    if "JSON_OUTPUT_START" in line:
        capture_json = True
        continue
    if "JSON_OUTPUT_END" in line:
        capture_json = False
        continue
    if capture_json:
        json_output_lines.append(line)

json_output = "\n".join(json_output_lines)
result_output = json.loads(json_output)
print(result_output)
print(type(result_output))

# for key in result_output.keys():
#     print(result_output[key])
#     print(type(result_output[key]))