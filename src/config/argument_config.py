from dataclasses import dataclass
@dataclass
class ArgumentConfig:
    server_port: int = 8080
    server_name: str = '127.0.0.1'
