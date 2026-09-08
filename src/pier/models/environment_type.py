from enum import Enum


class EnvironmentType(str, Enum):
    DOCKER = "docker"
    MODAL = "modal"
    DAYTONA = "daytona"
    E2B = "e2b"
