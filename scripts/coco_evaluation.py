from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.spice.spice import Spice
from pycocoevalcap.rouge.rouge import Rouge
import pickle
import numpy as np
import itertools
from bert_score import score

class SocraticEvalCap:
    def __init__(self, gts, res_raw):
        """
        Adapted from the COCOEvalCap class from pycocoevalcap/coco_evaluation.py.

        :param coco:
        :param cocoRes:
        """
        self.evalImgs = []
        self.eval = {}
        self.gts_sims = {}
        self.res_sims = {}
        self.imgToEval = {}

        self.res_cossim = res_raw

        #Make res a suitable format for the rule-based evaluation
        res = {}
        for i, row in res_raw.iterrows():
            res[row.image_name] = [{
                'image_id': row.image_name,
                'id': row.image_name,
                'caption': row.generated_caption
            }]

        self.intersect_keys = set(gts.keys()) & set(res.keys())
        gts = {key: gts.get(key) for key in self.intersect_keys}
        res = {key: res.get(key) for key in self.intersect_keys}

        self.res = res
        self.gts = gts
        self.img_ids = self.gts.keys()

    def evaluate_rulebased(self):
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


    def evaluate_cossim(self):
        # Get the clip embeddings for images and captions
        with open('../cache/embed_imgs.pickle', 'rb') as handle:
            embed_imgs = pickle.load(handle)

        with open('../cache/embed_capt_gt.pickle', 'rb') as handle:
            embed_capt_gt = pickle.load(handle)

        # with open(f'cache/embed_capt_res_{self.approach}.pickle', 'rb') as handle:
        #     embed_capt_res = pickle.load(handle)

        # Calculate similarities between images and captions
        for img_id in self.intersect_keys:
            # GT
            try:
                self.gts_sims[img_id] = (embed_capt_gt[img_id] @ embed_imgs[img_id].T).flatten().tolist()
            except:
                print('t')

            # RES
            # self.res_sims[img_id] = float(embed_capt_res[img_id] @ embed_imgs[img_id].T)

        gts_list = list(itertools.chain(*self.gts_sims.values()))
        # res_list = list(self.res_sims.values())

        # Calculate aggregates
        self.sims = {
            'gts': [np.mean(gts_list), np.std(gts_list)],
            'res': [self.res_cossim['cosine_similarity'].mean(), self.res_cossim['cosine_similarity'].std()]
        }


    def evaluate_bert(self):

        cands = []
        for image_id, caption_list in self.res.items():
            cpt_dict = caption_list[0]
            cands.append(cpt_dict['caption'])

        refs = []
        for image_id, caption_list in self.gts.items():
            current_ref = ''
            for cpt_dict in caption_list:
                current_ref += ' ' + cpt_dict['caption']
            refs.append(current_ref)

        P, R, F1 = score(cands, refs, lang="en", verbose=True)

        self.bert_scores = {
            'P': [P.mean(), P.std()],
            'R': [R.mean(), R.std()],
            'F1': [F1.mean(), F1.std()]
        }

