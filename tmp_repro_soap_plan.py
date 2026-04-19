from core.soap_engine import SoapEngine
engine = SoapEngine()
text = 'History of present illness: Patient reports knee pain. Plan of Action: Begin gentle strengthening. Functional & Diet Findings: Patient has high blood pressure. Exercise Session: Squat, lunge. Vitals: Weight 180 lbs, BP 130/80 mmHg, BMI 28. Tasks and Time Tracking: none.'
plan = engine.generate_plan_from_text(text)
print('profile session_duration:', plan['profile'].get('session_duration'))
print('profile days:', plan['profile'].get('days'))
print('profile weekly_days:', plan['profile'].get('weekly_days'))
print('plan days count:', len(plan['plan']))
for day, content in plan['plan'].items():
    print('DAY', day)
    print('  warmup len', len(content.get('warmup', [])))
    print('  cooldown len', len(content.get('cooldown', [])))
    print('  main len', len(content.get('main_workout', [])))
    print('  warmup', [x.get('name') for x in content.get('warmup', [])])
    print('  cooldown', [x.get('name') for x in content.get('cooldown', [])])
    print('  main', [x.get('name') for x in content.get('main_workout', [])])
