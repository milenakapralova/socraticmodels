from pycocotools.coco import COCO
from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.spice.spice import Spice
from pycocoevalcap.rouge.rouge import Rouge


class SocraticEvalCap:
    def __init__(self, gts, res):
        """
        Adapted from the COCOEvalCap class from pycocoevalcap/eval.py.

        gts[id] = [{
            'image_id': 391895,
            'id': 770337,
            'caption': 'A man with a red helmet on a small moped on a dirt road. '
        }, ...]
        res[id] = [{
            'image_id': 391895,
            'id': 770337,
            'caption': 'A man with a red helmet on a small moped. '
        }]
        :param coco:
        :param cocoRes:
        """
        self.evalImgs = []
        self.eval = {}
        self.imgToEval = {}

        union_keys = set(gts.keys()) & set(res.keys())
        gts = {key: gts.get(key) for key in union_keys}
        res = {key: res.get(key) for key in union_keys}

        self.res = res
        self.gts = gts
        self.img_ids = self.gts.keys()

    def evaluate(self):
        # imgIds = self.coco.getImgIds()
        # gts = {}
        # res = {}
        # for imgId in self.img_ids:
        #     gts[imgId] = self.gts[imgId]
        #     res[imgId] = self.res[imgId]

        # =================================================
        # Set up scorers
        # =================================================
        print('tokenization...')
        tokenizer = PTBTokenizer()
        gts_tok = tokenizer.tokenize(self.gts)
        res_tok = tokenizer.tokenize(self.res)
        # =================================================
        # Set up scorers
        # =================================================
        print('setting up scorers...')
        scorers = [
            (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
            (Meteor(),"METEOR"),
            (Rouge(), "ROUGE_L"),
            (Cider(), "CIDEr"),
            (Spice(), "SPICE")
        ]

        # =================================================
        # Compute scores
        # =================================================
        for scorer, method in scorers:
            print('computing %s score...'%(scorer.method()))
            score, scores = scorer.compute_score(gts_tok, res_tok)
            if type(method) == list:
                for sc, scs, m in zip(score, scores, method):
                    self.setEval(sc, m)
                    self.setImgToEvalImgs(scs, gts_tok.keys(), m)
                    print("%s: %0.3f"%(m, sc))
            else:
                self.setEval(score, method)
                self.setImgToEvalImgs(scores, gts_tok.keys(), method)
                print("%s: %0.3f"%(method, score))
        self.setEvalImgs()

    def setEval(self, score, method):
        self.eval[method] = score

    def setImgToEvalImgs(self, scores, imgIds, method):
        for imgId, score in zip(imgIds, scores):
            if not imgId in self.imgToEval:
                self.imgToEval[imgId] = {}
                self.imgToEval[imgId]["image_id"] = imgId
            self.imgToEval[imgId][method] = score

    def setEvalImgs(self):
        self.evalImgs = [eval for imgId, eval in self.imgToEval.items()]
