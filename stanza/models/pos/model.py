import logging
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence, pack_sequence, pad_sequence, PackedSequence

from stanza.models.common.bert_embedding import extract_bert_embeddings
from stanza.models.common.biaffine import BiaffineScorer
from stanza.models.common.foundation_cache import load_bert, load_charlm
from stanza.models.common.hlstm import HighwayLSTM
from stanza.models.common.dropout import WordDropout
from stanza.models.common.vocab import CompositeVocab
from stanza.models.common.char_model import CharacterModel
from stanza.models.constituency.positional_encoding import AddSinusoidalEncoding
# this is from a pip installable package, if we want to use something
# more stable than our own position encoding
# from positional_encodings import PositionalEncoding1D

logger = logging.getLogger('stanza')

class Tagger(nn.Module):
    def __init__(self, args, vocab, emb_matrix=None, share_hid=False, foundation_cache=None):
        super().__init__()

        self.vocab = vocab
        self.args = args
        self.share_hid = share_hid
        self.unsaved_modules = []

        def add_unsaved_module(name, module):
            self.unsaved_modules += [name]
            setattr(self, name, module)

        # input layers
        input_size = 0
        if self.args['word_emb_dim'] > 0:
            # frequent word embeddings
            self.word_emb = nn.Embedding(len(vocab['word']), self.args['word_emb_dim'], padding_idx=0)
            input_size += self.args['word_emb_dim']

        if not share_hid:
            # upos embeddings
            self.upos_emb = nn.Embedding(len(vocab['upos']), self.args['tag_emb_dim'], padding_idx=0)

        if self.args['char'] and self.args['char_emb_dim'] > 0:
            if self.args.get('charlm', None):
                if args['charlm_forward_file'] is None or not os.path.exists(args['charlm_forward_file']):
                    raise FileNotFoundError('Could not find forward character model: {}  Please specify with --charlm_forward_file'.format(args['charlm_forward_file']))
                if args['charlm_backward_file'] is None or not os.path.exists(args['charlm_backward_file']):
                    raise FileNotFoundError('Could not find backward character model: {}  Please specify with --charlm_backward_file'.format(args['charlm_backward_file']))
                logger.debug("POS model loading charmodels: %s and %s", args['charlm_forward_file'], args['charlm_backward_file'])
                add_unsaved_module('charmodel_forward', load_charlm(args['charlm_forward_file'], foundation_cache=foundation_cache))
                add_unsaved_module('charmodel_backward', load_charlm(args['charlm_backward_file'], foundation_cache=foundation_cache))
                input_size += self.charmodel_forward.hidden_dim() + self.charmodel_backward.hidden_dim()
            else:
                bidirectional = args.get('char_bidirectional', False)
                self.charmodel = CharacterModel(args, vocab, bidirectional=bidirectional)
                if bidirectional:
                    self.trans_char = nn.Linear(self.args['char_hidden_dim'] * 2, self.args['transformed_dim'], bias=False)
                else:
                    self.trans_char = nn.Linear(self.args['char_hidden_dim'], self.args['transformed_dim'], bias=False)
                input_size += self.args['transformed_dim']

        if self.args['bert_model']:
            bert_model, bert_tokenizer = load_bert(self.args['bert_model'], foundation_cache)
            input_size += bert_model.config.hidden_size
            if args.get('bert_hidden_layers', False):
                # The average will be offset by 1/N so that the default zeros
                # repressents an average of the N layers
                self.bert_layer_mix = nn.Linear(args['bert_hidden_layers'], 1, bias=False)
                nn.init.zeros_(self.bert_layer_mix.weight)
            else:
                # an average of layers 2, 3, 4 will be used
                # (for historic reasons)
                self.bert_layer_mix = None
        else:
            bert_model = None
            bert_tokenizer = None
        add_unsaved_module('bert_model', bert_model)
        add_unsaved_module('bert_tokenizer', bert_tokenizer)

        if self.args['pretrain']:
            # pretrained embeddings, by default this won't be saved into model file
            add_unsaved_module('pretrained_emb', nn.Embedding.from_pretrained(torch.from_numpy(emb_matrix), freeze=True))
            self.trans_pretrained = nn.Linear(emb_matrix.shape[1], self.args['transformed_dim'], bias=False)
            input_size += self.args['transformed_dim']

        self.mha = None
        if self.args.get('attention', False):
            #add_unsaved_module('position_encoding', PositionalEncoding1D(input_size))
            add_unsaved_module('position_encoding', AddSinusoidalEncoding(input_size))
            self.layer_norm = nn.LayerNorm(input_size)
            self.mha = nn.MultiheadAttention(input_size, self.args['attention_heads'], batch_first=True)
            # alternatives:
            #self.encoder_layer = nn.TransformerEncoderLayer(input_size, 1, input_size, batch_first=True)
            #self.transformer = nn.TransformerEncoder(encoder_layer, 1)
            # also, ideally we could put pattn or lattn here

        # recurrent layers
        self.taggerlstm = HighwayLSTM(input_size, self.args['hidden_dim'], self.args['num_layers'], batch_first=True, bidirectional=True, dropout=self.args['dropout'], rec_dropout=self.args['rec_dropout'], highway_func=torch.tanh)
        self.drop_replacement = nn.Parameter(torch.randn(input_size) / np.sqrt(input_size))
        self.taggerlstm_h_init = nn.Parameter(torch.zeros(2 * self.args['num_layers'], 1, self.args['hidden_dim']))
        self.taggerlstm_c_init = nn.Parameter(torch.zeros(2 * self.args['num_layers'], 1, self.args['hidden_dim']))

        # classifiers
        self.upos_hid = nn.Linear(self.args['hidden_dim'] * 2, self.args['deep_biaff_hidden_dim'])
        self.upos_clf = nn.Linear(self.args['deep_biaff_hidden_dim'], len(vocab['upos']))
        self.upos_clf.weight.data.zero_()
        self.upos_clf.bias.data.zero_()

        if share_hid:
            clf_constructor = lambda insize, outsize: nn.Linear(insize, outsize)
        else:
            self.xpos_hid = nn.Linear(self.args['hidden_dim'] * 2, self.args['deep_biaff_hidden_dim'] if not isinstance(vocab['xpos'], CompositeVocab) else self.args['composite_deep_biaff_hidden_dim'])
            self.ufeats_hid = nn.Linear(self.args['hidden_dim'] * 2, self.args['composite_deep_biaff_hidden_dim'])
            clf_constructor = lambda insize, outsize: BiaffineScorer(insize, self.args['tag_emb_dim'], outsize)

        if isinstance(vocab['xpos'], CompositeVocab):
            self.xpos_clf = nn.ModuleList()
            for l in vocab['xpos'].lens():
                self.xpos_clf.append(clf_constructor(self.args['composite_deep_biaff_hidden_dim'], l))
        else:
            self.xpos_clf = clf_constructor(self.args['deep_biaff_hidden_dim'], len(vocab['xpos']))
            if share_hid:
                self.xpos_clf.weight.data.zero_()
                self.xpos_clf.bias.data.zero_()

        self.ufeats_clf = nn.ModuleList()
        for l in vocab['feats'].lens():
            if share_hid:
                self.ufeats_clf.append(clf_constructor(self.args['deep_biaff_hidden_dim'], l))
                self.ufeats_clf[-1].weight.data.zero_()
                self.ufeats_clf[-1].bias.data.zero_()
            else:
                self.ufeats_clf.append(clf_constructor(self.args['composite_deep_biaff_hidden_dim'], l))

        # criterion
        self.crit = nn.CrossEntropyLoss(ignore_index=0) # ignore padding

        self.drop = nn.Dropout(args['dropout'])
        self.worddrop = WordDropout(args['word_dropout'])

    def log_norms(self):
        lines = ["NORMS FOR MODEL PARAMTERS"]
        for name, param in self.named_parameters():
            if param.requires_grad and name.split(".")[0] not in ('bert_model', 'charmodel_forward', 'charmodel_backward'):
                lines.append("  %s %.6g" % (name, torch.norm(param).item()))
        logger.info("\n".join(lines))

    def forward(self, word, word_mask, wordchars, wordchars_mask, upos, xpos, ufeats, pretrained, word_orig_idx, sentlens, wordlens, text):
        
        def pack(x):
            return pack_padded_sequence(x, sentlens, batch_first=True)

        inputs = []
        if self.args['word_emb_dim'] > 0:
            word_emb = self.word_emb(word)
            word_emb = pack(word_emb)
            inputs += [word_emb]

        if self.args['pretrain']:
            pretrained_emb = self.pretrained_emb(pretrained)
            pretrained_emb = self.trans_pretrained(pretrained_emb)
            pretrained_emb = pack(pretrained_emb)
            inputs += [pretrained_emb]

        def pad(x):
            return pad_packed_sequence(PackedSequence(x, word_emb.batch_sizes), batch_first=True)[0]

        if self.args['char'] and self.args['char_emb_dim'] > 0:
            if self.args.get('charlm', None):
                all_forward_chars = self.charmodel_forward.build_char_representation(text)
                all_forward_chars = pack(pad_sequence(all_forward_chars, batch_first=True))
                all_backward_chars = self.charmodel_backward.build_char_representation(text)
                all_backward_chars = pack(pad_sequence(all_backward_chars, batch_first=True))
                inputs += [all_forward_chars, all_backward_chars]
            else:
                char_reps = self.charmodel(wordchars, wordchars_mask, word_orig_idx, sentlens, wordlens)
                char_reps = PackedSequence(self.trans_char(self.drop(char_reps.data)), char_reps.batch_sizes)
                inputs += [char_reps]

        if self.bert_model is not None:
            device = next(self.parameters()).device
            processed_bert = extract_bert_embeddings(self.args['bert_model'], self.bert_tokenizer, self.bert_model, text, device, keep_endpoints=False,
                                                     num_layers=self.bert_layer_mix.in_features if self.bert_layer_mix is not None else None)

            if self.bert_layer_mix is not None:
                # add the average so that the default behavior is to
                # take an average of the N layers, and anything else
                # other than that needs to be learned
                # TODO: refactor this
                processed_bert = [self.bert_layer_mix(feature).squeeze(2) + feature.sum(axis=2) / self.bert_layer_mix.in_features for feature in processed_bert]

            processed_bert = pad_sequence(processed_bert, batch_first=True)
            inputs += [pack(processed_bert)]

        lstm_inputs = torch.cat([x.data for x in inputs], 1)
        lstm_inputs = self.worddrop(lstm_inputs, self.drop_replacement)
        lstm_inputs = self.drop(lstm_inputs)
        lstm_inputs = PackedSequence(lstm_inputs, inputs[0].batch_sizes)

        if self.mha is not None:
            orig_inputs, lstm_sizes = pad_packed_sequence(lstm_inputs, batch_first=True)
            # these print statements are super noisy, but if the
            # weights are miscalibrated, they show the model blowing
            # up almost immediately
            #print("orig input: %.4f" % torch.linalg.norm(orig_inputs), end="  ")
            # this would work if using PositionalEncoding1D from the positional_encodings package
            # pos_inputs = self.position_encoding(orig_inputs)
            # lstm_inputs = orig_inputs + pos_inputs * 0.1
            # higher POS factors than this just fail horribly
            lstm_inputs = self.position_encoding(orig_inputs, 0.1)
            #print("pos input: %.4f  total input: %.4f" % (torch.linalg.norm(lstm_inputs - orig_inputs), torch.linalg.norm(lstm_inputs)), end="  ")

            # build an attention mask as the batch may have differing lengths
            attn_mask = torch.zeros(lstm_inputs.shape[0], lstm_inputs.shape[1], dtype=torch.bool, device=lstm_inputs.device)
            for lstm_idx, lstm_size in enumerate(lstm_sizes):
                attn_mask[lstm_idx, lstm_size:] = True   # True should mean don't attend

            attn_outputs, attn_weights = self.mha(lstm_inputs, lstm_inputs, lstm_inputs, key_padding_mask=attn_mask)
            # multiplying by 0.02 is very unsatisfactory, though
            # it means very little derivative flows from the attention
            # to the layers below
            # not multiplying makes it blow up almost immediately
            # weirdly, initializing made no difference.  the problem occurs
            # almost immediately as the first updates cause the out_proj
            # to jump to a much larger value
            lstm_inputs = attn_outputs * 0.02
            # could do a residual link here
            #lstm_inputs = orig_inputs + lstm_inputs
            #print("  attn output: %.4f" % (torch.linalg.norm(lstm_inputs)))
            lstm_inputs = pack(lstm_inputs)

        lstm_outputs, _ = self.taggerlstm(lstm_inputs, sentlens, hx=(self.taggerlstm_h_init.expand(2 * self.args['num_layers'], word.size(0), self.args['hidden_dim']).contiguous(), self.taggerlstm_c_init.expand(2 * self.args['num_layers'], word.size(0), self.args['hidden_dim']).contiguous()))
        lstm_outputs = lstm_outputs.data

        upos_hid = F.relu(self.upos_hid(self.drop(lstm_outputs)))
        upos_pred = self.upos_clf(self.drop(upos_hid))

        preds = [pad(upos_pred).max(2)[1]]

        upos = pack(upos).data
        loss = self.crit(upos_pred.view(-1, upos_pred.size(-1)), upos.view(-1))

        if self.share_hid:
            xpos_hid = upos_hid
            ufeats_hid = upos_hid

            clffunc = lambda clf, hid: clf(self.drop(hid))
        else:
            xpos_hid = F.relu(self.xpos_hid(self.drop(lstm_outputs)))
            ufeats_hid = F.relu(self.ufeats_hid(self.drop(lstm_outputs)))

            if self.training:
                upos_emb = self.upos_emb(upos)
            else:
                upos_emb = self.upos_emb(upos_pred.max(1)[1])

            clffunc = lambda clf, hid: clf(self.drop(hid), self.drop(upos_emb))

        xpos = pack(xpos).data
        if isinstance(self.vocab['xpos'], CompositeVocab):
            xpos_preds = []
            for i in range(len(self.vocab['xpos'])):
                xpos_pred = clffunc(self.xpos_clf[i], xpos_hid)
                loss += self.crit(xpos_pred.view(-1, xpos_pred.size(-1)), xpos[:, i].view(-1))
                xpos_preds.append(pad(xpos_pred).max(2, keepdim=True)[1])
            preds.append(torch.cat(xpos_preds, 2))
        else:
            xpos_pred = clffunc(self.xpos_clf, xpos_hid)
            loss += self.crit(xpos_pred.view(-1, xpos_pred.size(-1)), xpos.view(-1))
            preds.append(pad(xpos_pred).max(2)[1])

        ufeats_preds = []
        ufeats = pack(ufeats).data
        for i in range(len(self.vocab['feats'])):
            ufeats_pred = clffunc(self.ufeats_clf[i], ufeats_hid)
            loss += self.crit(ufeats_pred.view(-1, ufeats_pred.size(-1)), ufeats[:, i].view(-1))
            ufeats_preds.append(pad(ufeats_pred).max(2, keepdim=True)[1])
        preds.append(torch.cat(ufeats_preds, 2))

        return loss, preds
