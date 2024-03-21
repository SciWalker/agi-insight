import subprocess
import json
import os
def run_ollama_in_docker(model_name,prompt):
    """
    Executes the specified ollama model within the Docker container.
    
    Parameters:
    - model_name (str): The name of the model to run.
    
    Returns:
    - output (str): The output from the ollama model.
    """
    docker_command = ["C:\\Program Files\\Docker\\Docker\\resources\\bin\\docker.exe", "exec", "-i", "ollama", "ollama", "run", model_name]
    process = subprocess.run(docker_command, capture_output=True, text=True, input=prompt)
    if process.returncode == 0:
        print("Command executed successfully.")
        return process.stdout
    else:
        print("Error executing command.")
        return process.stderr

def run_ollama_served_in_docker(model_name,prompt):
    docker_command = ["docker", "exec", "-i", "ollama", "ollama", "run", model_name]
    process = subprocess.run(docker_command, capture_output=True, text=True, input=prompt, encoding='utf-8')
    if process.returncode == 0:
        print("Command executed successfully.")
        return process.stdout
    else:
        print("Error executing command.")
        return process.stderr


def run_ollama_interactive(model_name):
    """
    Starts an interactive session with the specified ollama model within the Docker container.
    Continuously reads the output until a condition is met.
    """
    docker_command = ['C:\\Program Files\\Docker\\Docker\\resources\\bin\\docker.exe', 'exec', 'ollama', 'ollama', 'run', model_name]
    process = subprocess.Popen(docker_command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1)

    # Example of sending a command/query to the model
    query = "Hello, ollama model.\n"
    print(f"Sending query: {query}")
    process.stdin.write(query)
    process.stdin.flush()

    # Continuously read the output
    for line in iter(process.stdout.readline, ''):
        print("Response from ollama model:", line, end='')  # Use end='' to avoid adding extra newline
        if line == "\n" or "specific termination keyword" in line:  # Define your termination condition
            break
    process.terminate()  # Terminate the process when done
def test_docker_output():
    """
    Test capturing output from a Docker container with a simple command.
    """
    docker_command = ['C:\\Program Files\\Docker\\Docker\\resources\\bin\\docker.exe', 'exec', 'ollama', 'echo', '"Hello World"']
    process = subprocess.Popen(docker_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    stdout, stderr = process.communicate()

    if stderr:
        print("Error:", stderr)
    else:
        print("Output:", stdout)
# Example usage:
if __name__ == "__main__":
    model_name = "phi"  # Specify your model name
    command = ""  # Specify the command to run with ollama
    prompt = "create a poem about John Calvin and his influence on the Reformation."
    output = run_ollama_served_in_docker(model_name,prompt)
    #save output
    with open("output.txt", "w") as file:
        file.write(output)