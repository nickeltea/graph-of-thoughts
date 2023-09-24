import requests
import base64
import json
import csv

# Replace with your GitHub username and personal access token
username = "nickeltea"
access_token = "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"

# API endpoint for issues
issues_url = f"https://api.github.com/repos/zyedidia/micro/issues"

# Set up headers for authentication
headers = {
    "Authorization": f"Basic {base64.b64encode(f'{username}:{access_token}'.encode()).decode()}"
}

# Send GET request to fetch issues
response = requests.get(issues_url, headers=headers)

# Save issues to JSON file
def JSON_issues():
    output_filename = "extracted_issues.json"

    if response.status_code == 200:
        issues = response.json()
        
        with open(output_filename, "w") as json_file:
            json.dump(issues, json_file, indent=4)
            
        print(f"Issues extracted and saved to {output_filename}")
    else:
        print("Error fetching issues:", response.status_code)
        print(response.text)
    
    return output_filename

# Save issues to a CSV file
def CSV_issues():
    output_filename = "extracted_issues.csv"

    if response.status_code == 200:
        issues = response.json()
        
        with open(output_filename, "w", newline="", encoding="utf-8") as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(["Issue Number", "Title", "Description"])
            for issue in issues:
                csv_writer.writerow([issue["number"], issue["title"], issue.get("body", "")])
            
        print(f"Issues extracted and saved to {output_filename}")
    else:
        print("Error fetching issues:", response.status_code)
        print(response.text)
    
    return output_filename


