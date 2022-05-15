import torch
from torch import nn
import numpy as np
import torch.distributed as dist
from .image_encoder.visual_transformer import visual_transformer_B32
from .text_encoder.text_transformer import text_transformers
import os
import functools
BN = None
allreduce = dist.all_reduce
allgather = dist.all_gather
broadcast = dist.broadcast
synchronize = torch.cuda.synchronize
init_process_group = dist.init_process_group
allreduce_async = functools.partial(dist.all_reduce, async_op=True)

def get_rank():
    return int(os.environ.get('SLURM_PROCID', 0))

def get_world_size():
    return int(os.environ.get('SLURM_NTASKS', 1))

def barrier():
    if int(os.environ.get('SLURM_NTASKS', 1)) > 1:
        x = torch.cuda.IntTensor([1])
        dist.all_reduce(x)
        x.cpu()

class AllGather(torch.autograd.Function):

    @staticmethod
    def forward(ctx, tensor):
        ctx.rank = get_rank()
        ctx.world_size = get_world_size()
        y = [tensor.new(*tensor.size()) for _ in range(ctx.world_size)]
        allgather(y, tensor)
        y = torch.cat(y, 0).view(-1, *tensor.size())
        return y

    @staticmethod
    def backward(ctx, grad_output):
        in_grad = torch.zeros_like(grad_output)
        in_grad.copy_(grad_output)
        allreduce(in_grad)
        return in_grad[ctx.rank]

class IMTEPRO(nn.Module):
    def __init__(self,image_encode, text_encode, use_allgather):
        super().__init__()
        self.use_allgather = use_allgather
        self.visual =image_encode
        self.encode_text = text_encode
        self.logit_scale = nn.Parameter(torch.ones([1]))
        # self.logit_scale = nn.Parameter(torch.ones([]))
        nn.init.constant_(self.logit_scale, np.log(1/0.07))

    def text_parameters(self):
        param = [self.logit_scale]
        if self.encode_text.text_encode_type == 'Transformer':
            param.append(self.encode_text.positional_embedding)
        elif self.encode_text.text_encode_type == 'Bert':
            param.extend([self.encode_text.text_transformer.cls.predictions.bias])
        return param

    def text_modules(self):
        if self.encode_text.text_encode_type == 'Transformer':
            return [self.encode_text.transformer, self.encode_text.text_projection, self.encode_text.token_embedding, self.encode_text.ln_final]
        elif self.encode_text.text_encode_type == 'Bert':
            return [self.encode_text.text_transformer.bert, self.encode_text.text_projection,
                    self.encode_text.text_transformer.cls.predictions.transform]
        else:
            import ipdb
            ipdb.set_trace()
            return [self.encode_text.text_transformer, self.encode_text.text_projection]

    def visual_parameters(self):
        return []

    def visual_modules(self):
        return [self.visual]

    @property
    def dtype(self):
        try:
            return self.visual.conv1.weight.dtype
        except:
            try:
                return self.visual.head.weight.dtype
            except:
                try:
                    return self.visual.stem[0].weight.dtype
                except:
                    return self.encode_text.text_projection.weight.dtype

    def encode_image(self, image):
        return self.visual(image.type(self.dtype))

    def sample_captions(self, texts):
        return [text[0] for text in texts]

    def all_gather(self, input):
        output = AllGather.apply(input)
        output = output.view(-1, *(output.shape[2:]))
        return output

    def forward(self, input, all_gather=False):
        # input
        images = input['images']
        texts = input['captions']
        texts = self.sample_captions(texts)
        # text&image encode
        image_features = self.encode_image(images)
        text_features = self.encode_text(texts)


        # normalized features
        image_features = image_features / (image_features.norm(dim=-1, keepdim=True))
        text_features = text_features / (text_features.norm(dim=-1, keepdim=True)+1e-10)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logit_scale.data = torch.clamp(logit_scale.data, max=100)

        if self.training and self.use_allgather or all_gather:
            gathered_image_features = self.all_gather(image_features)
            gathered_text_features = self.all_gather(text_features)

            logits_per_image = logit_scale * image_features @ gathered_text_features.t()
            logits_per_text = logit_scale * text_features @ gathered_image_features.t()
        else:
            logits_per_image = logit_scale * image_features @ text_features.t()
            logits_per_text = logit_scale * text_features @ image_features.t()

        return logits_per_image, logits_per_text



class FOSTER(IMTEPRO):
    def __init__(self,image_encode, text_encode, use_allgather, \
                 return_dense=False, return_caption=False, text_mask_type=None, dense_mapping_image=2048, \
                 dense_mapping_language=512, dense_embed_dim=256, select_topk=False):
        super(FOSTER, self).__init__(image_encode, text_encode, use_allgather)
        self.return_dense = return_dense
        self.return_caption = return_caption

        self.text_mask_type = text_mask_type
        self.select_topk = select_topk
        if self.return_dense:
            self.image_mapping = nn.Linear(dense_mapping_image, dense_embed_dim)
            self.text_mapping = nn.Linear(dense_mapping_language, dense_embed_dim)

        self.logit_scale_dense = nn.Parameter(torch.ones([]))
        nn.init.constant_(self.logit_scale_dense, np.log(1/0.07))

        if self.encode_text.text_encode_type == 'Transformer':
            self.sos_index = self.encode_text.tokenizer.encoder["<|startoftext|>"]
            self.padding_idx = 0

        self.caption_module = None
        if text_mask_type is not None:
            enc_dim = self.encode_text.text_projection.weight.shape[-1]
            self.text_label_predictor = nn.Linear(enc_dim, self.encode_text.vocab_size)

    def encode_text_dense(self, texts, return_dense=True):
        text_features, word_features = self.encode_text(texts, return_dense=return_dense)
        word_features_d = self.text_mapping(word_features)
        return word_features_d

    def encode_image_dense(self, image):
        image_features, image_features_dense = self.visual(image.type(self.dtype), return_dense=True)
        image_features_dense = self.image_mapping(image_features_dense)
        return image_features_dense

    def encode_image(self, image, return_all=False):
        output = self.visual(image.type(self.dtype), return_dense=return_all)
        return output

    def get_weighted_dense_logits(self, dense_feat_1, dense_feat_2, top_k=16):
        dense_feat_1 = dense_feat_1 / dense_feat_1.norm(dim=-1, keepdim=True)
        dense_feat_2 = dense_feat_2 / dense_feat_2.norm(dim=-1, keepdim=True)

        logit_scale_dense = self.logit_scale_dense.exp()

        if self.select_topk:
            dense_feat_cross_logit = torch.matmul(dense_feat_1, dense_feat_2.permute(0, 2, 1))
            _, dense_id_1 = torch.topk(dense_feat_cross_logit.sum(dim=2), dim=1, k=top_k)
            _, dense_id_2 = torch.topk(dense_feat_cross_logit.sum(dim=1), dim=1, k=top_k)
            bs, n1 = dense_feat_1.shape[:2]
            dense_id_1 = dense_id_1 + (torch.arange(bs) * n1).to(dense_id_1.device)[:, None]
            selected_feat_1 = dense_feat_1.reshape(bs * n1, -1)[dense_id_1].reshape(bs, top_k, -1)
            bs, n2 = dense_feat_2.shape[:2]
            dense_id_2 = dense_id_2 + (torch.arange(bs) * n2).to(dense_id_2.device)[:, None]
            selected_feat_2 = dense_feat_2.reshape(bs * n2, -1)[dense_id_2].reshape(bs, top_k, -1)

        selected_feat_1 = dense_feat_1
        selected_feat_2 = dense_feat_2

        selected_feat_1 = self.all_gather(selected_feat_1)
        selected_feat_2 = self.all_gather(selected_feat_2)


        def get_logits(dense_feat_1, selected_feat_2):
            i, j, k = dense_feat_1.shape
            l, m, k = selected_feat_2.shape
            dense_feat_1 = dense_feat_1.reshape(-1, k)
            selected_feat_2 = selected_feat_2.reshape(-1, k)
            final_logits_1 = logit_scale_dense * dense_feat_1 @ selected_feat_2.t()
            final_logits_1 = final_logits_1.reshape(i, j, l, m).permute(0,2,1,3)
            return final_logits_1
        final_logits_1 = get_logits(dense_feat_1, selected_feat_2).max(dim=-1)[0].mean(dim=-1)
        final_logits_2 = get_logits(dense_feat_2, selected_feat_1).max(dim=-1)[0].mean(dim=-1)
        return final_logits_1, final_logits_2


    def forward(self, input, return_dict=False):
        # data input
        images = input['images']
        images_1, _ = torch.split(images, [3,3], dim=1)
        texts = input['captions']
        texts = self.sample_captions(texts)
        # text
        text_features, word_features, text_labels = self.encode_text(texts, mask_type = self.text_mask_type)
        # image
        image_concat = images_1
        image_features_1, image_features_d = self.encode_image(image_concat, return_all=True)
        logit_scale = self.logit_scale.exp()
        image_features_1 = image_features_1 / (image_features_1.norm(dim=-1, keepdim=True))
        text_features = text_features / (text_features.norm(dim=-1, keepdim=True)+1e-10)
        if self.training and self.use_allgather:
            barrier()
            gathered_image_features_1 = self.all_gather(image_features_1)
            gathered_text_features = self.all_gather(text_features)
            logits_per_image_1 = logit_scale * image_features_1 @ gathered_text_features.t()
            logits_per_text_1 = logit_scale * text_features @ gathered_image_features_1.t()
        if self.return_dense:
            image_features_d1 = image_features_d
            image_features_d1 = self.image_mapping(image_features_d1)
            word_features_d = self.text_mapping(word_features)
            logits_per_image_dense_1, logits_per_text_dense_1 = self.get_weighted_dense_logits(image_features_d1, word_features_d)
        if return_dict:
            ret_dict = {}
            ret_dict['logits'] = logits_per_image_1, logits_per_text_1
            if self.return_dense:
                ret_dict['dense_logits'] = logits_per_image_dense_1, logits_per_text_dense_1
            return ret_dict
        raise NotImplementedError()


def fosterer_vitb32(**kwargs):
    image_encode = visual_transformer_B32(**kwargs['image_encode'])
    text_encode = text_transformers(**kwargs['text_encode'])
    model = FOSTER(image_encode,text_encode,**kwargs['clip'], dense_mapping_image=768)
    return model
