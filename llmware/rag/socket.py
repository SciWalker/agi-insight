import socket

def connect_to_ollama(host='localhost', port=11434):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((host, port))
        # Example: send a command to the ollama service
        s.sendall(b'Hello, ollama!')
        # Receive the response
        data = s.recv(1024)
    print('Received:', data.decode())

if __name__ == "__main__":
    connect_to_ollama()