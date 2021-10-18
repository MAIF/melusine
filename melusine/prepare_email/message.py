class Message:
    def __init__(self, text, meta):
        self.text = text
        self.meta = meta

    def __repr__(self):
        x = "--------- Message ---------\n"
        x += f"Meta : {self.meta.strip()}\n"
        x += f"Text : {self.text.strip()}\n"
        x += "---------------------------"

        return x
