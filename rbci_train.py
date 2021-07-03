import tensorflow as tf
import numpy as np
import os
import sys
import getopt
import random
import datetime
import traceback
from ABCEI.rb_net import RBNet
from ABCEI.util import *
from sklearn import metrics

# Define parameter flags
flags = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('loss', 'l2', """Which loss function to use (l1/l2/log)""")
tf.app.flags.DEFINE_integer('n_in', 2, """Number of representation layers. """)
tf.app.flags.DEFINE_integer('n_out', 2, """Number of regression layers. """)
tf.app.flags.DEFINE_integer('n_dc', 2, """Number of discriminator layers. """)
tf.app.flags.DEFINE_float('p_lambda', 0.0, """Weight decay regularization parameter. """)
tf.app.flags.DEFINE_float('p_beta', 10.0, """Gradient penalty weight. """)
tf.app.flags.DEFINE_integer('rep_weight_decay', 1, """Whether to penalize representation layers with weight decay""")
tf.app.flags.DEFINE_float('dropout_in', 0.9, """Input layers dropout keep rate. """)
tf.app.flags.DEFINE_float('dropout_out', 0.9, """Output layers dropout keep rate. """)
tf.app.flags.DEFINE_string('nonlin', 'relu', """Kind of non-linearity. Default relu. """)
tf.app.flags.DEFINE_float('lrate', 0.05, """Learning rate. """)
tf.app.flags.DEFINE_float('decay', 0.5, """RMSProp decay. """)
tf.app.flags.DEFINE_integer('batch_size', 100, """Batch size. """)
tf.app.flags.DEFINE_integer('dim_in', 100, """Layer dimensions of Encoder network. """)
tf.app.flags.DEFINE_integer('dim_out', 100, """Post-representation layer dimensions. """)
tf.app.flags.DEFINE_integer('dim_mi', 100, """MI estimation layer dimensions. """)
tf.app.flags.DEFINE_integer('dim_d', 100, """Discriminator layer dimensions. """)
tf.app.flags.DEFINE_integer('batch_norm', 0, """Whether to use batch normalization. """)
tf.app.flags.DEFINE_string('normalization', 'none',
                           """How to normalize representation (after batch norm). none/bn_fixed/divide/project """)
tf.app.flags.DEFINE_integer('experiments', 1, """Number of experiments. """)
tf.app.flags.DEFINE_integer('iterations', 2000, """Number of iterations. """)
tf.app.flags.DEFINE_float('weight_init', 0.01, """Weight initialization scale. """)
tf.app.flags.DEFINE_float('lrate_decay', 0.95, """Decay of learning rate every 100 iterations """)
tf.app.flags.DEFINE_integer('varsel', 0, """Whether the first layer performs variable selection. """)
tf.app.flags.DEFINE_string('outdir', '../results/ihdp/', """Output directory. """)
tf.app.flags.DEFINE_string('datadir', '../data/topic/csv/', """Data directory. """)
tf.app.flags.DEFINE_string('dataform', 'topic_dmean_seed_%d.csv', """Training data filename form. """)
tf.app.flags.DEFINE_string('data_test', '', """Test data filename form. """)
tf.app.flags.DEFINE_integer('sparse', 0, """Whether data is stored in sparse format (.x, .y). """)
tf.app.flags.DEFINE_integer('seed', 1, """Seed. """)
tf.app.flags.DEFINE_integer('repetitions', 1, """Repetitions with different seed.""")
tf.app.flags.DEFINE_integer('use_p_correction', 1, """Whether to use population size p(t) in mmd/disc/wass.""")
tf.app.flags.DEFINE_string('optimizer', 'RMSProp', """Which optimizer to use. (RMSProp/Adagrad/GradientDescent/Adam)""")
tf.app.flags.DEFINE_integer('output_csv', 0, """Whether to save a CSV file with the results""")
tf.app.flags.DEFINE_integer('output_delay', 100, """Number of iterations between log/loss outputs. """)
tf.app.flags.DEFINE_integer('pred_output_delay', -1,
                            """Number of iterations between prediction outputs. (-1 gives no intermediate output). """)
tf.app.flags.DEFINE_integer('debug', 0, """Debug mode. """)
tf.app.flags.DEFINE_integer('save_rep', 0, """Save representations after training. """)
tf.app.flags.DEFINE_float('val_part', 0, """Validation part. """)
tf.app.flags.DEFINE_boolean('split_output', 0, """Whether to split output layers between treated and control. """)
tf.app.flags.DEFINE_boolean('reweight_sample', 1,
                            """Whether to reweight sample for prediction loss with average treatment probability. """)

if flags.sparse:
    import scipy.sparse as sparse

NUM_ITERATIONS_PER_DECAY = 100

__DEBUG__ = False
if flags.debug:
    __DEBUG__ = True


def train(rbnet, sess, train_step, train_discriminator_step, train_encoder_step, train_hrep_step, data_exp,
          valid_data_exp_id_arr,
          test_data_exp,
          logfile, i_exp):
    """ Trains a rbnet model on supplied data """

    ''' Train/validation split '''
    data_num = data_exp['x'].shape[0]
    range_of_data_num = range(data_num)
    train_index = list(set(range_of_data_num) - set(valid_data_exp_id_arr))
    train_num = len(train_index)

    ''' Compute treatment probability'''
    p_treated = np.mean(data_exp['t'][train_index, :])

    z_norm = np.random.normal(0., 1., (1, flags.dim_in))

    ''' Set up loss feed_dicts'''
    # todo dict_factual means in train_data
    dict_factual = {rbnet.x: data_exp['x'][train_index, :], rbnet.t: data_exp['t'][train_index, :],
                    rbnet.y_: data_exp['yf'][train_index, :],
                    rbnet.do_in: 1.0, rbnet.do_out: 1.0, rbnet.r_lambda: flags.p_lambda, rbnet.r_beta: flags.p_beta,
                    rbnet.p_t: p_treated, rbnet.z_norm: z_norm}

    if flags.val_part > 0:
        dict_valid = {rbnet.x: data_exp['x'][valid_data_exp_id_arr, :],
                      rbnet.t: data_exp['t'][valid_data_exp_id_arr, :],
                      rbnet.y_: data_exp['yf'][valid_data_exp_id_arr, :],
                      rbnet.do_in: 1.0, rbnet.do_out: 1.0, rbnet.r_lambda: flags.p_lambda, rbnet.r_beta: flags.p_beta,
                      rbnet.p_t: p_treated, rbnet.z_norm: z_norm}
    else:
        dict_valid = dict()

    if data_exp['HAVE_TRUTH']:
        dict_cfactual = {rbnet.x: data_exp['x'][train_index, :], rbnet.t: 1 - data_exp['t'][train_index, :],
                         rbnet.y_: data_exp['ycf'][train_index, :],
                         rbnet.do_in: 1.0, rbnet.do_out: 1.0, rbnet.z_norm: z_norm}
    else:
        dict_cfactual = dict()

    ''' Initialize TensorFlow variables '''
    sess.run(tf.global_variables_initializer())

    ''' Set up for storing predictions '''
    preds_train = []
    preds_test = []

    ''' Compute losses '''
    losses = []
    obj_loss, f_error, gmi_err, discriminator_loss, rep_loss, gradient_pen = \
        sess.run(
            [rbnet.tot_loss, rbnet.pred_loss, rbnet.gmi_neg_loss, rbnet.discriminator_loss, rbnet.rep_loss, rbnet.dp],
            feed_dict=dict_factual)

    cf_error = np.nan
    if data_exp['HAVE_TRUTH']:
        cf_error = sess.run(rbnet.pred_loss, feed_dict=dict_cfactual)

    if flags.val_part > 0:
        valid_obj, valid_f_error, valid_gmi, valid_dc, valid_rep_r, valid_dp = \
            sess.run([rbnet.tot_loss, rbnet.pred_loss, rbnet.gmi_neg_loss, rbnet.discriminator_loss, rbnet.rep_loss,
                      rbnet.dp],
                     feed_dict=dict_valid)
    else:
        valid_obj, valid_f_error, valid_gmi, valid_dc, valid_rep_r, valid_dp = np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

    losses.append([obj_loss, f_error, cf_error, gmi_err, discriminator_loss, rep_loss, gradient_pen,
                   valid_f_error, valid_gmi, valid_dc, valid_rep_r, valid_dp, valid_obj])

    objnan = False

    reps = []
    reps_test = []

    ''' Train for multiple iterations '''
    for i in range(flags.iterations):

        ''' Fetch sample '''
        # I = random.sample(range(0, train_num), flags.batch_size)
        # x_batch = data_exp['x'][train_index,:][I,:]
        # t_batch = data_exp['t'][train_index,:][I]
        # y_batch = data_exp['yf'][train_index,:][I]

        I = list(range(0, train_num))
        np.random.shuffle(I)
        for i_batch in range(train_num // flags.batch_size):
            if i_batch < train_num // flags.batch_size - 1:
                I_b = I[i_batch * flags.batch_size:(i_batch + 1) * flags.batch_size]
            else:
                I_b = I[i_batch * flags.batch_size:]
            x_batch = data_exp['x'][train_index, :][I_b, :]
            t_batch = data_exp['t'][train_index, :][I_b]
            y_batch = data_exp['yf'][train_index, :][I_b]

            z_norm_batch = np.random.normal(0., 1., (1, flags.dim_in))

            # if __DEBUG__:
            #     M = sess.run(rbnet.pop_dist(rbnet.x, rbnet.t), feed_dict={rbnet.x: x_batch, rbnet.t: t_batch})
            #     log(logfile, 'Median: %.4g, Mean: %.4f, Max: %.4f' % (
            #         np.median(M.tolist()), np.mean(M.tolist()), np.amax(M.tolist())))

            ''' Do one step of gradient descent '''
            if not objnan:
                sess.run(train_hrep_step, feed_dict={rbnet.x: x_batch,
                                                     rbnet.do_in: flags.dropout_in, rbnet.do_out: flags.dropout_out})

                # train discriminator
                for sub_dc in range(0, 3):
                    sess.run(train_discriminator_step,
                             feed_dict={rbnet.x: x_batch, rbnet.t: t_batch, rbnet.r_beta: flags.p_beta,
                                        rbnet.do_in: flags.dropout_in, rbnet.do_out: flags.dropout_out,
                                        rbnet.z_norm: z_norm_batch})
                # train encoder
                # for sub_enc in range(0,2):
                sess.run(train_encoder_step, feed_dict={rbnet.x: x_batch, rbnet.t: t_batch,
                                                        rbnet.do_in: flags.dropout_in, rbnet.do_out: flags.dropout_out,
                                                        rbnet.z_norm: z_norm_batch})

                sess.run(train_step, feed_dict={rbnet.x: x_batch, rbnet.t: t_batch,
                                                rbnet.y_: y_batch, rbnet.do_in: flags.dropout_in,
                                                rbnet.do_out: flags.dropout_out,
                                                rbnet.r_lambda: flags.p_lambda, rbnet.p_t: p_treated})

            ''' Project variable selection weights '''
            if flags.varsel:
                wip = simplex_project(sess.run(rbnet.weights_in[0]), 1)
                sess.run(rbnet.projection, feed_dict={rbnet.w_proj: wip})

        ''' Compute loss every N iterations '''
        if i % flags.output_delay == 0 or i == flags.iterations - 1:
            obj_loss, f_error, gmi_err, discriminator_loss, rep_loss, gradient_pen = \
                sess.run([rbnet.tot_loss, rbnet.pred_loss, rbnet.gmi_neg_loss, rbnet.discriminator_loss, rbnet.rep_loss,
                          rbnet.dp],
                         feed_dict=dict_factual)

            rep = sess.run(rbnet.h_rep_norm, feed_dict={rbnet.x: data_exp['x'], rbnet.do_in: 1.0})
            rep_norm = np.mean(np.sqrt(np.sum(np.square(rep), 1)))

            cf_error = np.nan
            if data_exp['HAVE_TRUTH']:
                cf_error = sess.run(rbnet.pred_loss, feed_dict=dict_cfactual)

            valid_obj = np.nan
            valid_imb = np.nan
            valid_f_error = np.nan
            if flags.val_part > 0:
                valid_obj, valid_f_error, valid_gmi, valid_dc, valid_rep_r, valid_dp = \
                    sess.run(
                        [rbnet.tot_loss, rbnet.pred_loss, rbnet.gmi_neg_loss, rbnet.discriminator_loss, rbnet.rep_loss,
                         rbnet.dp],
                        feed_dict=dict_valid)

            losses.append([obj_loss, f_error, cf_error, gmi_err, discriminator_loss, rep_loss, gradient_pen,
                           valid_f_error, valid_gmi, valid_dc, valid_rep_r, valid_dp, valid_obj])
            loss_str = str(
                i) + '\tObj: %.3f,\tF: %.3f,\tCf: %.3f, \tGMI: %.3f, \tdc_loss: %.3f, \trep_loss: %.3f, \tdp: %.3f, \tVal: %.3f, \tValGMI: %.3f, \tValdc: %.3f, \tValrep: %.3f, \tValdp: %.3f, \tValObj: %.2f' \
                       % (
                           obj_loss, f_error, cf_error, -gmi_err, discriminator_loss, rep_loss, gradient_pen,
                           valid_f_error,
                           -valid_gmi, valid_dc, valid_rep_r, valid_dp, valid_obj)
            log(logfile, loss_str)

            # if flags.loss == 'log':
            #     y_pred = sess.run(rbnet.output, feed_dict={rbnet.x: x_batch,
            #                                                rbnet.t: t_batch, rbnet.do_in: 1.0, rbnet.do_out: 1.0})
            #     # y_pred = 1.0*(y_pred > 0.5)
            #     # acc = 100*(1 - np.mean(np.abs(y_batch - y_pred)))
            #
            #     fpr, tpr, thresholds = metrics.roc_curve(y_batch, y_pred)
            #     auc = metrics.auc(fpr, tpr)
            #
            #     loss_str += ',\tAuc_batch: %.2f' % auc

            # log(logfile, loss_str)

            if np.isnan(obj_loss):
                log(logfile, 'Experiment %d: Objective is NaN. Skipping.' % i_exp)
                objnan = True

        ''' Compute predictions every M iterations '''
        if (flags.pred_output_delay > 0 and i % flags.pred_output_delay == 0) or i == flags.iterations - 1:

            y_pred_f = sess.run(rbnet.output, feed_dict={rbnet.x: data_exp['x'],
                                                         rbnet.t: data_exp['t'], rbnet.do_in: 1.0, rbnet.do_out: 1.0})
            y_pred_cf = sess.run(rbnet.output, feed_dict={rbnet.x: data_exp['x'],
                                                          rbnet.t: 1 - data_exp['t'], rbnet.do_in: 1.0,
                                                          rbnet.do_out: 1.0})
            preds_train.append(np.concatenate((y_pred_f, y_pred_cf), axis=1))

            # if flags.loss == 'log' and data_exp['HAVE_TRUTH']:
            #     fpr, tpr, thresholds = metrics.roc_curve(np.concatenate((data_exp['yf'], data_exp['ycf']), axis=0),
            #                                              np.concatenate((y_pred_f, y_pred_cf), axis=0))
            #     auc = metrics.auc(fpr, tpr)
            #     loss_str += ',\tAuc_train: %.2f' % auc

            if test_data_exp:
                y_pred_f_test = sess.run(rbnet.output, feed_dict={rbnet.x: test_data_exp['x'],
                                                                  rbnet.t: test_data_exp['t'], rbnet.do_in: 1.0,
                                                                  rbnet.do_out: 1.0})
                y_pred_cf_test = sess.run(rbnet.output, feed_dict={rbnet.x: test_data_exp['x'],
                                                                   rbnet.t: 1 - test_data_exp['t'], rbnet.do_in: 1.0,
                                                                   rbnet.do_out: 1.0})
                preds_test.append(np.concatenate((y_pred_f_test, y_pred_cf_test), axis=1))

                # if flags.loss == 'log' and data_exp['HAVE_TRUTH']:
                #     fpr, tpr, thresholds = metrics.roc_curve(
                #         np.concatenate((test_data_exp['yf'], test_data_exp['ycf']), axis=0),
                #         np.concatenate((y_pred_f_test, y_pred_cf_test), axis=0))
                #     auc = metrics.auc(fpr, tpr)
                #     loss_str += ',\tAuc_test: %.2f' % auc

            if flags.save_rep and i_exp == 1:
                reps_i = sess.run([rbnet.h_rep], feed_dict={rbnet.x: data_exp['x'],
                                                            rbnet.do_in: 1.0, rbnet.do_out: 0.0})
                reps.append(reps_i)

                if test_data_exp:
                    reps_test_i = sess.run([rbnet.h_rep], feed_dict={rbnet.x: test_data_exp['x'],
                                                                     rbnet.do_in: 1.0, rbnet.do_out: 0.0})
                    reps_test.append(reps_test_i)

    return losses, preds_train, preds_test, reps, reps_test


def run(outdir):
    """ Runs an experiment and stores result in outdir """

    ''' Set up paths and start log '''
    npzfile = os.path.join(outdir, 'result')
    npzfile_test = os.path.join(outdir, 'result.test')
    repfile = os.path.join(outdir, 'reps')
    repfile_test = os.path.join(outdir, 'reps.test')
    outform = os.path.join(outdir, 'y_pred')
    outform_test = os.path.join(outdir, 'y_pred.test')
    lossform = os.path.join(outdir, 'loss')
    logfile = os.path.join(outdir, 'log.txt')
    f = open(logfile, 'w')
    f.close()
    dataform = os.path.join(flags.datadir, flags.dataform)
    dataform_test = os.path.join(flags.datadir, flags.data_test)

    ''' Set random seeds '''
    random.seed(flags.seed)
    tf.set_random_seed(flags.seed)
    np.random.seed(flags.seed)

    ''' Save parameters '''
    save_config(os.path.join(outdir, 'config.txt'))
    log(logfile, 'Training with hyperparameters: beta={:.2g}, lambda={:.2g}'.format(flags.p_beta, flags.p_lambda))

    ''' Load Data '''
    datapath = dataform
    datapath_test = dataform_test
    log(logfile, 'Train data:{}'.format(datapath))
    log(logfile, 'Test data:{}'.format(datapath_test))

    data = load_data(datapath)
    test_data = load_data(datapath_test)

    log(logfile, 'Loaded data with shape [{},{}]'.format(data['n'], data['dim']))

    ''' Start Session '''
    sess = tf.Session()

    ''' Initialize input placeholders '''
    x = tf.placeholder("float", shape=[None, data['dim']], name='x')  # Features
    t = tf.placeholder("float", shape=[None, 1], name='t')  # Treatent
    y_ = tf.placeholder("float", shape=[None, 1], name='y_')  # Outcome

    # todo maybe
    znorm = tf.placeholder("float", shape=[None, flags.dim_in], name='z_norm')

    ''' Parameter placeholders '''
    # r_lambda is coefficient of regularization of prediction network.
    r_lambda = tf.placeholder("float", name='r_lambda')
    # r_beta is coefficient of gradient penalty in GAN
    r_beta = tf.placeholder("float", name='r_beta')

    do_in = tf.placeholder("float", name='dropout_in')
    do_out = tf.placeholder("float", name='dropout_out')
    # treatment probability in all observations
    p = tf.placeholder("float", name='p_treated')

    ''' Define model graph '''
    log(logfile, 'Defining graph...\n')
    dims = [data['dim'], flags.dim_in, flags.dim_out, flags.dim_mi, flags.dim_d]

    rbnet = RBNet(x, t, y_, p, znorm, flags, r_lambda, r_beta, do_in, do_out, dims)

    ''' Set up optimizer '''
    global_step = tf.Variable(0, trainable=False)
    lr = tf.train.exponential_decay(flags.lrate, global_step,
                                    NUM_ITERATIONS_PER_DECAY, flags.lrate_decay, staircase=True)

    lr_gan = 5e-5

    counter_enc = tf.Variable(0, trainable=False)
    lr_enc = tf.train.exponential_decay(lr_gan, counter_enc,
                                        NUM_ITERATIONS_PER_DECAY, flags.lrate_decay, staircase=True)

    counter_dc = tf.Variable(0, trainable=False)
    lr_dc = tf.train.exponential_decay(lr_gan, counter_dc,
                                       NUM_ITERATIONS_PER_DECAY, flags.lrate_decay, staircase=True)

    counter_gmi = tf.Variable(0, trainable=False)
    lr_gmi = tf.train.exponential_decay(flags.lrate, counter_gmi,
                                        NUM_ITERATIONS_PER_DECAY, flags.lrate_decay, staircase=True)

    # if flags.optimizer == 'Adagrad':
    #     opt = tf.train.AdagradOptimizer(lr)
    # elif flags.optimizer == 'GradientDescent':
    #     opt = tf.train.GradientDescentOptimizer(lr)
    if flags.optimizer == 'Adam':
        opt = tf.train.AdamOptimizer(lr)
        opt_enc = tf.train.AdamOptimizer(
            learning_rate=lr_enc,
            beta1=0.5,
            beta2=0.9)
        opt_dc = tf.train.AdamOptimizer(
            learning_rate=lr_dc,
            beta1=0.5,
            beta2=0.9)
        opt_gmi = tf.train.AdamOptimizer(lr_gmi)
    else:
        opt = tf.train.RMSPropOptimizer(lr_gan)
        opt_enc = tf.train.RMSPropOptimizer(lr_gan)
        opt_dc = tf.train.RMSPropOptimizer(lr_gan)
        opt_gmi = tf.train.RMSPropOptimizer(lr_gan)

    ''' Unused gradient clipping '''
    # gvs = opt.compute_gradients(rbnet.tot_loss)
    # capped_gvs = [(tf.clip_by_value(grad, -1.0, 1.0), var) for grad, var in gvs]
    # train_step = opt.apply_gradients(capped_gvs, global_step=global_step)

    # var_scope_get
    var_enc = tf.get_collection(
        tf.GraphKeys.TRAINABLE_VARIABLES, scope='encoder')
    var_gmi = tf.get_collection(
        tf.GraphKeys.TRAINABLE_VARIABLES, scope='gmi')
    var_gmi.extend(var_enc)
    var_dc = tf.get_collection(
        tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator')
    var_pred = tf.get_collection(
        tf.GraphKeys.TRAINABLE_VARIABLES, scope='pred')
    var_pred.extend(var_enc)

    print("var_enc:", [v.name for v in var_enc])
    print()
    print("var_gmi:", [v.name for v in var_gmi])
    print()
    print("var_dc:", [v.name for v in var_dc])
    print()
    print("var_pred:", [v.name for v in var_pred])
    print()
    # todo why global_step is counter_gmi?
    train_hrep_step = opt_gmi.minimize(rbnet.gmi_neg_loss, global_step=counter_gmi, var_list=var_gmi)
    train_discriminator_step = opt_dc.minimize(rbnet.discriminator_loss, global_step=counter_dc, var_list=var_dc)
    train_encoder_step = opt_enc.minimize(rbnet.rep_loss, global_step=counter_enc, var_list=var_enc)
    # todo why train_step using var_pred(pred and enc)?
    train_step = opt.minimize(rbnet.tot_loss, global_step=global_step, var_list=var_pred)

    ''' Set up for saving variables '''
    all_losses = []
    all_preds_train = []
    all_preds_test = []
    all_valid = []
    if flags.varsel:
        all_weights = None
        all_beta = None

    ''' Handle repetitions '''
    n_experiments = flags.experiments
    if flags.repetitions > 1:
        if flags.experiments > 1:
            log(logfile, 'ERROR: Use of both repetitions and multiple experiments is currently not supported.')
            sys.exit(1)
        n_experiments = flags.repetitions

    ''' Run for all repeated experiments '''
    data_exp = dict()
    test_data_exp = dict()
    for i_exp in range(1, n_experiments + 1):
        log(logfile, 'Training on experiment {}/{}...'.format(i_exp, n_experiments))

        ''' Load Data (if multiple repetitions, reuse first set)'''
        if i_exp == 1 or flags.experiments > 1:
            data_exp['x'] = data['x'][:, :, i_exp - 1]
            data_exp['t'] = data['t'][:, i_exp - 1:i_exp]
            data_exp['yf'] = data['yf'][:, i_exp - 1:i_exp]
            if data['HAVE_TRUTH']:
                data_exp['ycf'] = data['ycf'][:, i_exp - 1:i_exp]
            else:
                data_exp['ycf'] = None

            test_data_exp['x'] = test_data['x'][:, :, i_exp - 1]
            test_data_exp['t'] = test_data['t'][:, i_exp - 1:i_exp]
            test_data_exp['yf'] = test_data['yf'][:, i_exp - 1:i_exp]
            if test_data['HAVE_TRUTH']:
                test_data_exp['ycf'] = test_data['ycf'][:, i_exp - 1:i_exp]
            else:
                test_data_exp['ycf'] = None

            data_exp['HAVE_TRUTH'] = data['HAVE_TRUTH']
            test_data_exp['HAVE_TRUTH'] = test_data['HAVE_TRUTH']

        ''' Split into training and validation sets '''
        train_data_exp_id_arr, valid_data_exp_id_arr = validation_split(data_exp, flags.val_part)

        ''' Run training loop '''
        losses, preds_train, preds_test, reps, reps_test = \
            train(rbnet, sess, train_step, train_discriminator_step, train_encoder_step, train_hrep_step, data_exp,
                  valid_data_exp_id_arr,
                  test_data_exp, logfile, i_exp)

        ''' Collect all reps '''
        all_preds_train.append(preds_train)
        all_preds_test.append(preds_test)
        all_losses.append(losses)

        ''' Fix shape for output (n_units, dim, n_reps, n_outputs) '''
        out_preds_train = np.swapaxes(np.swapaxes(all_preds_train, 1, 3), 0, 2)
        out_preds_test = np.swapaxes(np.swapaxes(all_preds_test, 1, 3), 0, 2)
        # print(all_losses)
        out_losses = np.swapaxes(np.swapaxes(all_losses, 0, 2), 0, 1)

        ''' Store predictions '''
        log(logfile, 'Saving result to {}...\n'.format(outdir))
        if flags.output_csv:
            np.savetxt(fname='{}_{}.csv'.format(outform, i_exp), X=preds_train[-1], delimiter=',')
            np.savetxt(fname='{}_{}.csv'.format(outform_test, i_exp), X=preds_test[-1], delimiter=',')
            np.savetxt(fname='{}_{}.csv'.format(lossform, i_exp), X=losses, delimiter=',')

        # ''' Compute weights if doing variable selection '''
        # if flags.varsel:
        #     if i_exp == 1:
        #         all_weights = sess.run(rbnet.weights_in[0])
        #         all_beta = sess.run(rbnet.weights_pred)
        #     else:
        #         all_weights = np.dstack((all_weights, sess.run(rbnet.weights_in[0])))
        #         all_beta = np.dstack((all_beta, sess.run(rbnet.weights_pred)))

        ''' Save results and predictions '''
        all_valid.append(valid_data_exp_id_arr)

        np.savez(npzfile, pred=out_preds_train, loss=out_losses, val=np.array(all_valid))

        np.savez(npzfile_test, pred=out_preds_test)

        ''' Save representations '''
        if flags.save_rep and i_exp == 1:
            np.savez(repfile, rep=reps)

            np.savez(repfile_test, rep=reps_test)


def main(argv=None):  # pylint: disable=unused-argument
    """ Main entry point """
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S-%f")
    outdir = os.path.join(flags.outdir, 'results_' + timestamp)
    os.mkdir(outdir)

    try:
        run(outdir)
    except Exception as e:
        with open(outdir + 'error.txt', 'w') as errfile:
            errfile.write(''.join(traceback.format_exception(*sys.exc_info())))
        raise


if __name__ == '__main__':
    tf.app.run()
