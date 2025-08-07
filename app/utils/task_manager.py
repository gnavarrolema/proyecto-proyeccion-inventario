import threading
import time
import uuid
from typing import Any, Dict

# Almacén global de tareas
task_store = {}


class TaskStatus:
    """Clase para manejar el estado de tareas asincrónicas."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class Task:
    """Representa una tarea asincrónica."""

    def __init__(self, task_id=None):
        self.id = task_id or str(uuid.uuid4())
        self.status = TaskStatus.PENDING
        self.progress = 0
        self.result = None
        self.error = None
        self.created_at = time.time()
        self.completed_at = None

    def update(self, status=None, progress=None, result=None, error=None):
        """Actualiza el estado de la tarea."""
        if status:
            self.status = status
            if status == TaskStatus.COMPLETED or status == TaskStatus.FAILED:
                self.completed_at = time.time()

        if progress is not None:
            self.progress = progress

        if result is not None:
            self.result = result

        if error is not None:
            self.error = error

    def to_dict(self):
        """Convierte la tarea a un diccionario para ser devuelto como JSON."""
        return {
            "id": self.id,
            "status": self.status,
            "progress": self.progress,
            "result": self.result,
            "error": self.error,
            "created_at": self.created_at,
            "completed_at": self.completed_at,
            "duration": (self.completed_at or time.time()) - self.created_at,
        }


def run_task_in_background(func, task_id, *args, **kwargs):
    """
    Ejecuta una función en un hilo separado y actualiza el estado de la tarea.

    Args:
        func: Función a ejecutar
        task_id: ID de la tarea
        args, kwargs: Argumentos para la función
    """
    task = Task(task_id)
    task_store[task_id] = task

    def wrapper():
        try:
            # Actualizar estado a 'running'
            task.update(status=TaskStatus.RUNNING)

            # Ejecutar función
            result = func(*args, **kwargs)

            # Actualizar estado a 'completed'
            task.update(
                status=TaskStatus.COMPLETED, progress=100, result=result
            )
        except Exception as e:
            # Actualizar estado a 'failed'
            import traceback

            error_details = str(e) + "\n" + traceback.format_exc()
            task.update(
                status=TaskStatus.FAILED,
                error=str(e),
                result={"error_details": error_details},
            )

    # Iniciar hilo
    thread = threading.Thread(target=wrapper)
    thread.daemon = True
    thread.start()

    return task_id


def get_task(task_id):
    """Obtiene el estado actual de una tarea."""
    task = task_store.get(task_id)
    return task.to_dict() if task else None
