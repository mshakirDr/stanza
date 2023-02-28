import stanza
from stanza.protobuf import SsurgeonRequest, SsurgeonResponse
from stanza.server.java_protobuf_requests import send_request, add_token, add_word_to_graph, JavaProtobufContext

SSURGEON_JAVA = "edu.stanford.nlp.semgraph.semgrex.ssurgeon.ProcessSsurgeonRequest"

class SsurgeonEdit:
    def __init__(self, semgrex_pattern, ssurgeon_edits, ssurgeon_id=None, notes=None):
        # not a named tuple so we can have defaults without requiring a python upgrade
        self.semgrex_pattern = semgrex_pattern
        self.ssurgeon_edits = ssurgeon_edits
        self.ssurgeon_id = ssurgeon_id
        self.notes = notes

def send_ssurgeon_request(request):
    return send_request(request, SsurgeonResponse, SSURGEON_JAVA)

def build_request(doc, ssurgeon_edits):
    request = SsurgeonRequest()

    for ssurgeon in ssurgeon_edits:
        ssurgeon_proto = request.ssurgeon.add()
        ssurgeon_proto.semgrex = ssurgeon.semgrex_pattern
        for operation in ssurgeon.ssurgeon_edits:
            ssurgeon_proto.operation.append(operation)
        if ssurgeon.ssurgeon_id is not None:
            ssurgeon_proto.id = ssurgeon.ssurgeon_id
        if ssurgeon.notes is not None:
            ssurgeon_proto.notes = ssurgeon.notes

    for sent_idx, sentence in enumerate(doc.sentences):
        graph = request.graph.add()
        word_idx = 0
        for token in sentence.tokens:
            for word in token.words:
                add_token(graph.token, word, token)
                add_word_to_graph(graph, word, sent_idx, word_idx)

                word_idx = word_idx + 1

    return request

def build_request_one_operation(doc, semgrex_pattern, ssurgeon_edits, ssurgeon_id=None, notes=None):
    ssurgeon_edit = SsurgeonEdit(semgrex_pattern, ssurgeon_edits, ssurgeon_id, notes)
    return build_request(doc, [ssurgeon_edit])

def process_doc(doc, ssurgeon_edits):
    """
    Returns the result of processing the given semgrex expression and ssurgeon edits on the stanza doc.

    Currently the return is a SsurgeonResponse from CoreNLP.proto
    """
    request = build_request(doc, ssurgeon_edits)

    return send_ssurgeon_request(request)

def process_doc_one_operation(doc, semgrex_pattern, ssurgeon_edits, ssurgeon_id=None, notes=None):
    request = build_request_one_operation(doc, semgrex_pattern, ssurgeon_edits, ssurgeon_id, notes)

    return send_ssurgeon_request(request)

class Ssurgeon(JavaProtobufContext):
    """
    Ssurgeon context window

    This is a context window which keeps a process open.  Should allow
    for multiple requests without launching new java processes each time.
    """
    def __init__(self, classpath=None):
        super(Ssurgeon, self).__init__(classpath, SsurgeonResponse, SSURGEON_JAVA)

    def process(self, doc, ssurgeon_edits):
        """
        Apply each of the ssurgeon patterns to each of the dependency trees in doc
        """
        request = build_request(doc, ssurgeon_edits)
        return self.process_request(request)

    def process_one_operation(self, doc, semgrex_pattern, ssurgeon_edits, ssurgeon_id=None, notes=None):
        """
        Convenience method - build one operation, then apply it
        """
        request = build_request_one_operation(doc, semgrex_pattern, ssurgeon_edits, ssurgeon_id, notes)
        return self.process_request(request)

def main():
    nlp = stanza.Pipeline('en',
                          processors='tokenize,pos,lemma,depparse')

    doc = nlp('Uro ruined modern.  Fortunately, Wotc banned him.')
    #print(doc.sentences[0].dependencies)
    print(doc)
    print(process_doc_one_operation(doc, "{}=source >obj=zzz {}=target", ["addEdge -gov source -dep target -reln iobj"]))

if __name__ == '__main__':
    main()
