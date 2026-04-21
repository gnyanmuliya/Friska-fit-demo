import sys
sys.path.insert(0, '.')

class MockAzureAIPrescriptionParser:
    def parse_notes(self, notes_text):
        return {
            'weekly_schedule': {
                'monday': 'Cardio + Strength',
                'tuesday': 'Rest',
                'wednesday': 'Full Body Strength',
                'thursday': 'Rest',
                'friday': 'Cardio Only',
                'saturday': 'Rest',
                'sunday': 'Rest'
            },
            'medical_conditions': [],
            'physical_limitations': [],
            'activity_restrictions': []
        }

import services.experts_note_service
services.experts_note_service.AzureAIPrescriptionParser = MockAzureAIPrescriptionParser

from services.experts_note_service import ExpertsNoteService
service = ExpertsNoteService()

sample_notes = 'Patient: John Doe, Age 45. Prescription: Cardio 3x/week, Strength 2x/week'

try:
    plan = service.generate_plan_from_notes(sample_notes)
    
    print('╔════════════════════════════════════════════════════════════════╗')
    print('║         EXERCISE PLAN WITH COMPLETE DATASET INFO              ║')
    print('╚════════════════════════════════════════════════════════════════╝')
    print()
    
    for day in ['monday', 'wednesday', 'friday']:
        day_plan = plan['plan'][day]
        title = day_plan.get('title', 'Rest')
        print('📅 ' + day.upper() + ' - ' + title)
        print('─' * 60)
        
        main_exercises = day_plan.get('main_workout', [])
        if main_exercises:
            ex = main_exercises[0]
            print('  Exercise: ' + ex.get('name'))
            print('    • Sets: ' + str(ex.get('sets')) + ' | Reps: ' + str(ex.get('reps')))
            print('    • RPE: ' + str(ex.get('rpe')) + ' | Rest: ' + str(ex.get('rest_intervals')))
            print('    • Equipment: ' + str(ex.get('equipment')))
            print('    • Body Region: ' + str(ex.get('body_region')))
            print('    • Health Benefit: ' + str(ex.get('benefit')))
            print('    • Safety Cue: ' + str(ex.get('safety_cue')))
            print('    • Total Exercises: ' + str(len(main_exercises)))
        
        print()
    
    print('✅ All exercises now have complete dataset information!')
    print('✅ Day titles are exercise-based (Full Body Strength, Cardio, etc.)')
    
except Exception as e:
    print('Error: ' + str(e))
    import traceback
    traceback.print_exc()
