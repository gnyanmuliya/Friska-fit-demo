from core.soap_engine import SoapEngine

def process_soap_note(text: str):
    engine = SoapEngine()
    return engine.generate_plan_from_text(text)
