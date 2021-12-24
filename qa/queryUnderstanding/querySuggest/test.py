import sys
import os
import argparse
import tensorflow as tf
import numpy as np

from qa.queryUnderstanding.querySuggest.modeling import GroverConfig, sample
from qa.queryUnderstanding.querySuggest import tokenization

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.DEBUG)
from tensorflow.python.util import deprecation

deprecation._PRINT_DEPRECATION_WARNINGS = False
try:
    from tensorflow.python.util import module_wrapper as deprecation
except ImportError:
    from tensorflow.python.util import deprecation_wrapper as deprecation

deprecation._PER_MODULE_WARNING_LIMIT = 0

parser = argparse.ArgumentParser(
    description=
    'Contextual generation (aka given some metadata we will generate articles')

parser.add_argument(
    '-input',
    dest='input',
    type=str,
    help='Text to complete',
)
parser.add_argument(
    '-model_config_fn',
    dest='model_config_fn',
    default='qa/queryUnderstanding/querySuggest/mega.json',
    type=str,
    help='Configuration JSON for the model',
)
parser.add_argument(
    '-model_ckpt',
    dest='model_ckpt',
    default='models/Query2Query/model.ckpt-850000',
    type=str,
    help='checkpoint file for the model',
)
parser.add_argument(
    '-batch_size',
    dest='batch_size',
    default=1,
    type=int,
    help=
    'How many things to generate per context. will split into chunks if need be',
)

parser.add_argument(
    '-max_batch_size',
    dest='max_batch_size',
    default=None,
    type=int,
    help=
    'max batch size. You can leave this out and we will infer one based on the number of hidden layers',
)
parser.add_argument(
    '-top_p',
    dest='top_p',
    default=5.0,
    type=float,
    help=
    'p to use for top p sampling. if this isn\'t none, use this for everthing')
parser.add_argument(
    '-min_len',
    dest='min_len',
    default=5,
    type=int,
    help='min length of sample',
)
parser.add_argument(
    '-eos_token',
    dest='eos_token',
    default=102,
    type=int,
    help='eos token id',
)
parser.add_argument(
    '-samples',
    dest='samples',
    default=5,
    type=int,
    help='num_samples',
)

parser.add_argument(
    '-do_topk',
    dest='do_topk',
    default=True,
    type=bool,
    help='do topk',
)


def extract_generated_target(output_tokens, tokenizer):
    """
    Given some tokens that were generated, extract the target
    :param output_tokens: [num_tokens] thing that was generated
    :param encoder: how they were encoded
    :param target: the piece of metadata we wanted to generate!
    :return:
    """
    # Filter out first instance of start token
    assert output_tokens.ndim == 1

    start_ind = 0
    end_ind = output_tokens.shape[0]

    return {
        'extraction':
        tokenization.printable_text(''.join(
            tokenizer.convert_ids_to_tokens(output_tokens))),
        'start_ind':
        start_ind,
        'end_ind':
        end_ind,
    }


args = parser.parse_args()
proj_root_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
vocab_file_path = "qa/queryUnderstanding/querySuggest/bert-base-chinese-vocab.txt"
tokenizer = tokenization.FullTokenizer(vocab_file=vocab_file_path,
                                       do_lower_case=True)
news_config = GroverConfig.from_json_file(args.model_config_fn)

# We might have to split the batch into multiple chunks if the batch size is too large
default_mbs = {12: 32, 24: 16, 48: 3}
max_batch_size = args.max_batch_size if args.max_batch_size is not None else default_mbs[
    news_config.num_hidden_layers]

# factorize args.batch_size = (num_chunks * batch_size_per_chunk) s.t. batch_size_per_chunk < max_batch_size
num_chunks = int(np.ceil(args.batch_size / max_batch_size))
batch_size_per_chunk = int(np.ceil(args.batch_size / num_chunks))

# This controls the top p for each generation.
top_p = np.ones(
    (num_chunks, batch_size_per_chunk), dtype=np.float32) * args.top_p

tf_config = tf.ConfigProto(allow_soft_placement=True)

with tf.Session(config=tf_config, graph=tf.Graph()) as sess:
    initial_context = tf.placeholder(tf.int32, [batch_size_per_chunk, None])
    p_for_topp = tf.placeholder(tf.float32, [batch_size_per_chunk])
    eos_token = tf.placeholder(tf.int32, [])
    min_len = tf.placeholder(tf.int32, [])
    tokens, probs = sample(news_config=news_config,
                           initial_context=initial_context,
                           eos_token=eos_token,
                           min_len=min_len,
                           ignore_ids=None,
                           p_for_topp=p_for_topp,
                           do_topk=args.do_topk)

    saver = tf.train.Saver()
    saver.restore(sess, args.model_ckpt)
    print('🍺Model loaded. ....')
    test = [
        '狗狗容易感染什么疾病', '哈士奇老拆家怎么办', '犬瘟热', '狗发烧', '金毛', '拉布拉多不吃东西怎么办',
        '犬细小病毒的症状', '犬细小', '我和的'
    ]
    for text in test:
        for i in range(args.samples):
            line = tokenization.convert_to_unicode(text)
            bert_tokens = tokenizer.tokenize(line)
            bert_tokens.append("[SEP]")
            encoded = tokenizer.convert_tokens_to_ids(bert_tokens)
            context_formatted = []
            context_formatted.extend(encoded)
            # Format context end
            gens = []
            gens_raw = []
            gen_probs = []
            for chunk_i in range(num_chunks):
                tokens_out, probs_out = sess.run(
                    [tokens, probs],
                    feed_dict={
                        initial_context:
                        [context_formatted] * batch_size_per_chunk,
                        eos_token: args.eos_token,
                        min_len: args.min_len,
                        p_for_topp: top_p[chunk_i]
                    })

                for t_i, p_i in zip(tokens_out, probs_out):
                    extraction = extract_generated_target(output_tokens=t_i,
                                                          tokenizer=tokenizer)
                    gens.append(extraction['extraction'])
            l = gens[0].replace('[UNK]', '').replace('##', '').split("[SEP]")
            print("Sample,", i + 1, " of ", args.samples, "source:", text,
                  "generate query:", l[1])
