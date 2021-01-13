import torch

def create_model(opt):
    if opt.model == 'FormattingModel':
        from .formatting_model import FormattingModel, InferenceModel
        if opt.isTrain:
            model = FormattingModel()
        else:
            model = InferenceModel()
    else:
        from .ui_model import UIModel
        model = UIModel()
    model.initialize(opt)
    if opt.verbose:
        print("model [%s] was created" % (model.name()))

    if opt.isTrain and len(opt.gpu_ids):
        model = torch.nn.DataParallel(model, device_ids=opt.gpu_ids)

    return model
