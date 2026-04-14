import asyncio
from core.SOAPnotetest import ClinicalExtractionTool, PrescriptionParserTool

def process_soap_note(text: str):
    tool = ClinicalExtractionTool()

    extracted = tool.parse_sections(text)

    parser = PrescriptionParserTool()

    result = asyncio.run(
        parser.execute(
            text,
            {
                "age": 49,
                "primary_goal": "Recovery"
            }
        )
    )

    return result