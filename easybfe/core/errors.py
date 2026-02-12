class ResourceConflictError(Exception):
    def __init__(self, message, payload=None):
        super().__init__(message)
        self.payload = payload
        self.status_code = 409

class ResourceNotFoundError(Exception):
    def __init__(self, message, payload=None):
        super().__init__(message)
        self.payload = payload
        self.status_code = 404