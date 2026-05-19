class TaskService:
    def __init__(self, repo: TaskRepository):
        self.repo = repo

    def create_task(self, task_data: TaskCreate):
        if task_data.due_date and task_data.due_date < datetime.utcnow():
            raise InvalidTaskException()

        return self.repo.create(task_data)
