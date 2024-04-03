import pixeltable.env as env
from datatransfer.remote import Remote
from pixeltable import Table


class LabelStudioProject(Remote):

    def __init__(
            self,
            project_id: int
    ):
        self.ls = env.Env.get().label_studio_client
        self.project_id = project_id
        self.project = self.ls.get_project(project_id)

    def push(self, t: Table) -> None:
        for row in t.collect():
            self.project.import_tasks([row])
