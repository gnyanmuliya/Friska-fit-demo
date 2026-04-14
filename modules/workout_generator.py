from core.fitness import FitnessDataset, ExerciseFilter, WorkoutPlanner, ExerciseSelector, WorkoutComposer

def generate_workout_plan(profile: dict):
    df = FitnessDataset.load()

    filtered = ExerciseFilter.apply_filters(df, profile)

    planner = WorkoutPlanner()
    selector = ExerciseSelector()
    composer = WorkoutComposer(selector)

    # VERY IMPORTANT → You may already have this logic inside fitness.py
    # If yes, call that main function instead

    plan = {
        "message": "Hook your existing planner function here"
    }

    return plan