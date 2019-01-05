from app import app, db
from app.models import Word, SubgraphemeFrequency, SubphonemeFrequency, FasttextVector, FasttextNeighbor

@app.shell_context_processor
def make_shell_context():
    return {'db': db,  'Word': Word, 'SubgraphemeFrequency': SubgraphemeFrequency, 'SubphonemeFrequency': SubphonemeFrequency, 'FasttextVector': FasttextVector, 'FasttextNeighbor': FasttextNeighbor}
