from argparse import ArgumentParser

BATCHNORM_MOMENTUM = 0.01

class Config(object):
    """Wrapper class for model hyperparameters."""

    def __init__(self):
        """
        Defaults
        """
        self.obj_mem_weight_type = None
        self.rel_mem_weight_type = None
        self.lambda_con = None
        self.rel_unc = None
        self.obj_unc = None
        self.no_logging = None
        self.obj_con_loss = None
        self.mlm = None
        self.eos_coef = None
        self.tracking = None
        self.K = None
        self.rel_head = None
        self.obj_head = None
        self.mem_feat_selection = None
        self.mem_fusion = None
        self.take_obj_mem_feat = None
        self.obj_mem_compute = None
        self.mode = None
        self.save_path = None
        self.model_path = None
        self.data_path = None
        self.datasize = None
        self.ckpt = None
        self.optimizer = None
        self.bce_loss = None
        self.lr = 1e-5
        self.enc_layer = 1  # 空间编码器层数
        self.dec_layer = 3  # 时间解码器层数
        self.nepoch = 10
        self.parser = self.setup_parser()
        self.args = vars(self.parser.parse_args())  # 把所有命令行参数存入 self.args，方便后续以字典方式使用
        self.__dict__.update(self.args)
        
        if self.mem_feat_lambda is not None:
            self.mem_feat_lambda = float(self.mem_feat_lambda)
        
        
        if self.rel_mem_compute == 'None' :
            self.rel_mem_compute = None
        if self.obj_loss_weighting == 'None':
            self.obj_loss_weighting = None
        if self.rel_loss_weighting == 'None':
            self.rel_loss_weighting = None

    def setup_parser(self):
        """
        Sets up an argument parser
        :return:
        """
        parser = ArgumentParser(description='training code')
        parser.add_argument('-mode', dest='mode', help='predcls/sgcls/sgdet', default='predcls', type=str)
        parser.add_argument('-save_path', default='output/', type=str)
        parser.add_argument('-model_path', default='YOUR_PATH_HERE/models/best_Mrecall_model.tar', type=str)
        parser.add_argument('-data_path', default='YOUR_PATH_HERE/datasets/ag/', type=str)
        parser.add_argument('-datasize', dest='datasize', help='mini dataset or whole', default='large', type=str)
        parser.add_argument('-ckpt', dest='ckpt', help='checkpoint', default=None, type=str)
        parser.add_argument('-optimizer', help='adamw/adam/sgd', default='adamw', type=str)
        parser.add_argument('-lr', dest='lr', help='learning rate', default=1e-5, type=float)
        parser.add_argument('-nepoch', help='epoch number', default=10, type=int)
        parser.add_argument('-enc_layer', dest='enc_layer', help='spatial encoder layer', default=1, type=int)
        parser.add_argument('-dec_layer', dest='dec_layer', help='temporal decoder layer', default=3, type=int)

        #logging arguments
        parser.add_argument('-log_iter', default=100, type=int)
        parser.add_argument('-no_logging', action='store_true')

        # frequency
        parser.add_argument('-freq', default=True, type=bool, help='frequency of training')

        # heads arguments
        parser.add_argument('-obj_head', default='linear', type=str, help='classification head type')
        parser.add_argument('-rel_head', default='bayesian', type=str, help='classification head type')
        parser.add_argument('-K', default=6, type=int, help='number of mixture models')

        # tracking arguments
        parser.add_argument('-tracking', action='store_true')

        # memory arguments
        parser.add_argument('-rel_mem_compute', default='joint', type=str, help='compute relation memory hallucination [seperate/joint/None]')
        parser.add_argument('-obj_mem_compute', action='store_true')
        parser.add_argument('-take_obj_mem_feat', action='store_true')
        parser.add_argument('-obj_mem_weight_type', default='simple', type=str, help='type of memory [both/al/ep/simple]')
        parser.add_argument('-rel_mem_weight_type', default='simple', type=str, help='type of memory [both/al/ep/simple]')
        parser.add_argument('-mem_fusion', default='late', type=str, help='early/late')
        parser.add_argument('-mem_feat_selection', default='manual', type=str, help='manual/automated')
        parser.add_argument('-mem_feat_lambda', default=0.5, type=str, help='selection lambda')
        parser.add_argument('-pseudo_thresh', default=7, type=int, help='pseudo label threshold')

        # uncertainty arguments
        parser.add_argument('-obj_unc', action='store_true')
        parser.add_argument('-rel_unc', action='store_true')

        #loss arguments
        parser.add_argument('-obj_loss_weighting', default=None, type=str, help='ep/al/None')
        parser.add_argument('-rel_loss_weighting', default=None, type=str, help='ep/al/None')
        parser.add_argument('-mlm', action='store_true')
        parser.add_argument('-eos_coef', default=1, type=float,help='background class scaling in ce or nll loss')
        parser.add_argument('-obj_con_loss', default=None, type=str,  help='intra video visual consistency loss for objects (euc_con/info_nce)')
        parser.add_argument('-lambda_con', default=1, type=float,help='visual consistency loss coef')

        # visualization args
        parser.add_argument('-vis', action='store_true', help='启用可视化输出')
        parser.add_argument('-vis_topk', default=-1, type=int, help='每帧显示前K个关系 (topK by score)')
        parser.add_argument('-vis_score_thresh', default=0.2, type=float, help='关系得分阈值 (过滤低分)')
        parser.add_argument('-vis_out_dir', default='vis', type=str, help='可视化子目录名')
        return parser
