from core.soap_engine import SoapEngine
engine = SoapEngine()
text = 'History of present illness: Patient reports knee pain. Plan of Action: Begin gentle strengthening. Functional & Diet Findings: Patient has high blood pressure. Exercise Session: Squat, lunge. Vitals: Weight 180 lbs, BP 130/80 mmHg, BMI 28. Tasks and Time Tracking: none.'
parsed = engine.parse_text(text)
plan = engine.generate_plan_from_text(text)
print('fitness rows=', len(engine.fitness_df))
print('soap rows=', len(engine.soap_df))
print('history=', parsed.history)
print('findings=', parsed.findings)
print('exercise_session=', parsed.exercise_session)
print('vitals=', parsed.structured_vitals)
print('prescribed_exercises=', parsed.prescribed_exercises)
print('plan keys=', list(plan.keys()))
print('plan nonempty=', bool(plan.get('plan')))
