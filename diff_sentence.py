from gensim.models.word2vec import *
from extract_features_by_lm import LanguageModelDualDynamicRnnTest
import logging
import gensim
import sys
import spacy


logger = logging.getLogger(__name__)
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
LOG_HANDLERS = [logging.FileHandler('../log/filter_sentences.log', 'w+', 'utf-8')]
logging.basicConfig(handlers=LOG_HANDLERS, level=logging.WARNING, format=LOG_FORMAT)
logging.getLogger('gensim').setLevel(logging.WARNING)

WMD_THRESHOLD = 2.56
# SPACY_SIM_THRESHOLD = 0.95
LM_RATIO_PROBA = 0.9
MAX_TOKEN_LEN = 40

# nlp = spacy.load('en')
w2v_model = gensim.models.KeyedVectors.load_word2vec_format('../model/word_embeddings')
logger.debug(f'w2v model vocab size = {len(w2v_model.vocab)}')
logger.info('Word2vec model is loaded.')
lm_path = '/home/pubsrv/data/tingxun/seq2word/model/single_dual/model.ckpt-1400416'
lm_config_path = '/home/pubsrv/data/tingxun/seq2word/gaoxin_lm.cfg'
vocab_path = '/home/pubsrv/data/tingxun/seq2word/data/en_vocab_lm'
language_model = LanguageModelDualDynamicRnnTest(lm_path, vocab_path, lm_config_path, batch_size=1)
logger.info('Language model is loaded.')


def recover_unk(sentence):
    unk_buffer = []
    output_buffer = []
    i = 0
    tokens = sentence.strip().split()
    while i < len(tokens):
        if tokens[i] == '<u>':
            i += 1
            while i < len(tokens) and tokens[i] != '</u>':
                unk_buffer.append(tokens[i])
                i += 1
            if i < len(tokens) and tokens[i] == '</u>':
                i += 1
            output_buffer.append(''.join(unk_buffer))
            unk_buffer = []
        else:
            output_buffer.append(tokens[i])
            i += 1
    return output_buffer


def filter_sentence_pairs(prefix):
    with open(f'../data/{prefix}.learner', 'r', encoding='utf-8') as ori_learner_file, \
            open(f'../data/{prefix}.native', 'r', encoding='utf-8') as ori_native_file, \
            open(f'../data/{prefix}.filtered.learner', 'w+', encoding='utf-8') as filtered_learner_file, \
            open(f'../data/{prefix}.filtered.native', 'w+', encoding='utf-8') as filtered_native_file:
        learner_buf = []
        native_buf = []
        for (ori_learner_line, ori_native_line) in zip(ori_learner_file, ori_native_file):
            learner_line = ' '.join(ori_learner_line.strip().split()[::-1])
            native_line = ori_native_line.strip()
            logger.debug('=' * 80)
            ori_learner_tokens = recover_unk(learner_line)
            ori_native_tokens = recover_unk(native_line)
            logger.debug(f"learner line: {' '.join(ori_learner_tokens)}")
            logger.debug(f"native  line: {' '.join(ori_native_tokens)}")
            num_learner_oov_words = sum([1 if t not in w2v_model.vocab else 0 for t in ori_learner_tokens])
            num_native_oov_words = sum([1 if t not in w2v_model.vocab else 0 for t in ori_native_tokens])
            wmd = w2v_model.wmdistance(ori_learner_tokens, ori_native_tokens)
            logger.debug(f'WMD = {wmd}')
            # sim = nlp(' '.join(ori_learner_tokens)).similarity(nlp(' '.join(ori_native_tokens)))
            # logger.debug(f'similarity = {sim}')
            # if wmd > WMD_THRESHOLD and sim < SPACY_SIM_THRESHOLD:
            if wmd > WMD_THRESHOLD:
                logger.debug(f'WMD is too big and similarity is too small, '
                             f'two sentences are talking about different things, skipped')
                continue
            if len(ori_learner_tokens) <= MAX_TOKEN_LEN and len(ori_native_tokens) <= MAX_TOKEN_LEN and \
                    num_learner_oov_words == num_native_oov_words and wmd > 0.:
                ori_learner_proba = language_model.quick_get_avg_log_prob(ori_learner_tokens)
                ori_native_proba = language_model.quick_get_avg_log_prob(ori_native_tokens)
                proba_ratio = ori_native_proba / ori_learner_proba
                logger.debug(f'learner sentence proba = {ori_learner_proba},'
                             f' native sentence proba = {ori_native_proba}, ratio = {proba_ratio}')
                if proba_ratio > LM_RATIO_PROBA and abs(proba_ratio - 1.) > 1.e-5:
                    logger.debug(f'No obvious improvement found, consider original sentence follows grammar rules.')
                else:
                    learner_buf.append(ori_learner_line)
                    native_buf.append(ori_native_line)
            else:
                learner_buf.append(ori_learner_line)
                native_buf.append(ori_native_line)
            if len(learner_buf) == 10000:
                filtered_learner_file.writelines(learner_buf)
                filtered_native_file.writelines(native_buf)
                learner_buf = []
                native_buf = []

        filtered_learner_file.writelines(learner_buf)
        filtered_native_file.writelines(native_buf)

filter_sentence_pairs(sys.argv[1])

