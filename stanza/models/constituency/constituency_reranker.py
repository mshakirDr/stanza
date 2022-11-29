from stanza.models.constituency.reranker import Reranker

class ConstituencyReranker(Reranker):
    def __init__(self, model, batch_size):
        # TODO: make it so we don't need some bizarre circular imports
        self.model = model
        self.batch_size = batch_size

    def score_trees(self, trees):
        if self.model.reverse_sentence:
            trees = [x.reverse() for x in trees]
        analyzed = self.model.analyze_trees(trees, self.batch_size, keep_state=False, keep_constituents=False, keep_scores=True)
        # TODO: nograd, maybe take the item?
        scores = [x.predictions[0].score for x in analyzed]
        return scores

    def score_parse_results(self, parse_results):
        trees = [x.predictions[0].tree for x in parse_results]
        scores = self.score_trees(trees)

        new_results = []
        for score, result in zip(scores, parse_results):
            new_predictions = list(result.predictions)
            new_predictions[0] = new_predictions[0]._replace(score=score)
            new_result = result._replace(predictions=new_predictions)
            new_results.append(new_result)
        return new_results
