import pandas as pd

class VideoMapper:
    def __init__(self, csv_path="dataset/Exercise videos.csv"):
        self.df = pd.read_csv(csv_path)

    def get_video(self, guid):
        if not guid:
            return None

        row = self.df[self.df["guidid"] == guid]
        if row.empty:
            return None

        return row.iloc[0]["video_url"]

    def enrich_plan(self, plan):
        """
        Adds video_url to each exercise in the plan using guidid
        """

        for day in plan:
            if "exercises" not in day:
                continue

            for exercise in day["exercises"]:
                guid = exercise.get("guidid") or exercise.get("guid")
                video_url = self.get_video(guid)
                exercise["video_url"] = video_url

        return plan