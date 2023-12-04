import torch
from transformers import BatchEncoding
import torch.nn.functional as F
from transformers import BertModel
import torch.nn.functional as F
from enum import Enum
import scipy.stats as st
import numpy as np
import matplotlib.pyplot as plt
from typing import Union, Iterable

class NormType(Enum):
    Linf = 0
    L2 = 1

class MirrorDescentAttacker:
    def __init__(
        self, 
        ref_net, 
        tokenizer, 
        epsilon, 
        num_iters,
        norm_type=NormType.Linf, 
        random_init=True, 
        cls_text=True
        cls_image=True, 
        *args, 
        **kwargs
    ):

        self.ref_net = ref_net
        self.tokenizer = tokenizer
        self.cls_text = cls_text
        self.norm_type = norm_type
        self.random_init = random_init
        self.epsilon = epsilon
        self.cls_image = cls_image
        self.preprocess = kwargs.get('preprocess')
        self.bounding = kwargs.get('bounding')
        if self.bounding is None:
            self.bounding = (0, 1)
        self.epsilon_per_iter = self.epsilon / num_iters * 1.25
        self.criterion = torch.nn.CosineSimilarity()

    def loss(self, image, text, adv_image, adv_text):
        return 0.5 * self.criterion(adv_image, text) + 0.5 * self.criterion(adv_image, text)

    def image_input_diversity(self, image):
        return image
    

    def image_step(self, image): # , num_iters
        if self.random_init:
            self.delta = random_init(image, self.norm_type, self.epsilon)
        else:
            self.delta = torch.zeros_like(image)

        if hasattr(self, 'kernel'):
            self.kernel = self.kernel.to(image.device)

        if hasattr(self, 'grad'):
            self.grad = torch.zeros_like(image)


        # epsilon_per_iter = self.epsilon / num_iters * 1.25

        self.delta = self.delta.detach()
        self.delta.requires_grad = True

        image_diversity = self.image_input_diversity(image + self.delta)
        #plt.imshow(image_diversity.cpu().detach().numpy()[0].transpose(1, 2, 0))
        if self.preprocess is not None:
            image_diversity = self.preprocess(image_diversity)

        yield image_diversity

        grad = self.get_grad()
        grad = self.image_normalize(grad)
        self.delta = self.delta.data + self.epsilon_per_iter * grad

        # constraint 1: epsilon
        self.delta = self.project(self.delta, self.epsilon)
        # constraint 2: image range
        self.delta = torch.clamp(image + self.delta, *self.bounding) - image

        yield (image + self.delta).detach()

    # image
    def get_grad(self):
        self.grad = self.delta.grad.clone()
        return self.grad

    # image
    def project(self, delta, epsilon):
        if self.norm_type == NormType.Linf:
            return torch.clamp(delta, -epsilon, epsilon)
        elif self.norm_type == NormType.L2:
            return clamp_by_l2(delta, epsilon)
        
    # image
    def image_normalize(self, grad):
        if self.norm_type == NormType.Linf:
            return torch.sign(grad)
        elif self.norm_type == NormType.L2:
            return grad / torch.norm(grad, dim=(1, 2, 3), p=2, keepdim=True)
        
    @staticmethod
    def zero_indices(x, top_k):
        return torch.topk(x, top_k, largeset=False, sorted=False).indices
    
    def truncate(self, x, top_k):
        indices = self.zero_indices(x.abs(), top_k)
        x = x.scatter_(index=indices, value=0.)
        return x
        
    def text_step(self, P, W, lr, top_k, P_grad, W_grad):
        P_ = P - lr * P_grad
        P_ = P_ / torch.norm(P_)
        P_ = self.truncate(P_, top_k) # 1
        P_ = self.truncate(P_, top_k).bool().float() # 2
        W_= W * torch.exp(-lr * W_grad)
        W = W_ / torch.norm(W)
        return P, W
    
    # @staticmethod
    # def one_hot(voc, keys: Union[str, Iterable]):
    #     if isinstance(keys, str):
    #         keys = [keys]
    #     return F.one_hot(torch.tensor(voc(keys)), num_classes=len(voc))
    
    def adversarial_text(self, V, W, W0, P):
        return V * (W0 * (torch.eye(V.shape[-1]) - torch.diag(self.truncate(P))) + W * torch.diag(self.truncate(P)))
    
    def run(self, image, text, adv, num_iters=10, k=10, max_length=30, alpha=3.0):
        with torch.no_grad():
            device = image.device
            text_input = self.tokenizer(text * self.repeat, padding='max_length', truncation=True, max_length=max_length,
                                        return_tensors="pt").to(device)
            origin_output = self.net.inference(self.image_normalize(image).repeat(self.repeat, 1, 1,1),
                                               text_input, use_embeds=False)
            if self.cls:
                origin_embeds = origin_output['fusion_output'][:, 0, :].detach()
            else:
                origin_embeds = origin_output['fusion_output'].flatten(1).detach()
            
            V = self.ref_net.embeddings.word_embeddings.weight
            W0 = torch.seros()
            W = F.one_hot(torch.arange(0, V.shape[-1]) % W0.shape[0]).requires_grad_
            P = torch.zeros(requires_grad=True)
            P[0] = 1.
            if self.random_init:
                self.delta = random_init(image, self.norm_type, self.epsilon)
            else:
                self.delta = torch.zeros_like(image)

            if hasattr(self, 'kernel'):
                self.kernel = self.kernel.to(image.device)

            if hasattr(self, 'grad'):
                self.grad = torch.zeros_like(image)

        for i in range(num_iters):
            if i == 0:
                l = self.loss(image, text, image + self.delta, self.adversarial_text(V, W, W0, P))
            else:
                l = self.loss(image, text, image_adv, self.adversarial_text(V, W, W0, P))
            l.backward()
            P, W = self.text_step(P, W, lr=0.003, top_k=1, P_grad=P.grad, W_grad=W.grad)
            text_adv = self.adversarial_text(V, W, W0, P)
            l.backward()
            image_adv = self.image_step(image)       
        return image_adv, text_adv



        # image_attack = self.image_attacker.attack(images, num_iters)
        # with torch.no_grad():
        #     text_adv = self.text_attacker.attack(self.net, images, text, k)
        #     text_input = self.tokenizer(text_adv * self.repeat, padding='max_length', truncation=True, max_length=max_length,
        #                                 return_tensors="pt").to(device)
        #     text_adv_output = self.net.inference(self.image_normalize(images).repeat(self.repeat, 1, 1, 1),
        #                                             text_input, use_embeds=False)
        #     if self.cls:
        #         text_adv_embed = text_adv_output['fusion_output'][:, 0, :].detach()
        #     else:
        #         text_adv_embed = text_adv_output['fusion_output'].flatten(1).detach()

        # for i in range(num_iters):
        #     text = self.text_step(text)
        #     image_diversity = next(image_attack)
        #     adv_output = self.net.inference(image_diversity, text_input, use_embeds=False)
        #     if self.cls:
        #         adv_embed = adv_output['fusion_output'][:, 0, :]
        #     else:
        #         adv_embed = adv_output['fusion_output'].flatten(1)
        #     loss_clean_text = criterion(adv_embed.log_softmax(dim=-1), origin_embeds.softmax(dim=-1))

        #     loss_adv_text = criterion(adv_embed.log_softmax(dim=-1), text_adv_embed.softmax(dim=-1))
        #     # FORMULA 6
        #     loss = loss_adv_text + alpha * loss_clean_text
        #     loss.backward()
        # images_adv = next(image_attack)

        # # text
        # # ?????
        # # FORMULA 3
        # if adv == 1 or adv == 3 or adv == 4 or adv == 5:
        #     with torch.no_grad():
        #         text_adv = self.text_attacker.attack(self.net, images, text, k)
        # else:
        #     text_adv = text

        # return images_adv, text_adv



    # def run_trades(self, net, image, num_iters):
    #     with torch.no_grad():
    #         origin_output = net.inference_image(self.preprocess(image))
    #         if self.cls:
    #             origin_embed = origin_output['image_embed'][:, 0, :].detach()
    #         else:
    #             origin_embed = origin_output['image_embed'].flatten(1).detach()

    #     criterion = torch.nn.KLDivLoss(reduction='batchmean')
    #     attacker = self.attack(image, num_iters)

    #     for i in range(num_iters):
    #         image_adv = next(attacker)
    #         adv_output = net.inference_image(image_adv)
    #         if self.cls:
    #             adv_embed = adv_output['image_embed'][:, 0, :]
    #         else:
    #             adv_embed = adv_output['image_embed'].flatten(1)

    #         loss = criterion(adv_embed.log_softmax(dim=-1), origin_embed.softmax(dim=-1))
    #         loss.backward()

    #     image_adv = next(attacker)
    #     return image_adv


def equal_normalize(x):
    return x

class NormType(Enum):
    Linf = 0
    L2 = 1

def clamp_by_l2(x, max_norm):
    norm = torch.norm(x, dim=(1,2,3), p=2, keepdim=True)
    factor = torch.min(max_norm / norm, torch.ones_like(norm))
    return x * factor

def random_init(x, norm_type, epsilon):
    delta = torch.zeros_like(x)
    if norm_type == NormType.Linf:
        delta.data.uniform_(0.0, 1.0)
        delta.data = delta.data * epsilon
    elif norm_type == NormType.L2:
        delta.data.uniform_(0.0, 1.0)
        delta.data = delta.data - x
        delta.data = clamp_by_l2(delta.data, epsilon)
    return delta

def gkern(kernlen=21, nsig=3):
  """Returns a 2D Gaussian kernel array."""
  x = np.linspace(-nsig, nsig, kernlen)
  kern1d = st.norm.pdf(x)
  kernel_raw = np.outer(kern1d, kern1d)
  kernel = kernel_raw / kernel_raw.sum()
  return kernel


class ImageAttacker():
    # PGD
    def __init__(self, epsilon, norm_type=NormType.Linf, random_init=True, cls=True, *args, **kwargs):
        self.norm_type = norm_type
        self.random_init = random_init
        self.epsilon = epsilon
        self.cls = cls
        self.preprocess = kwargs.get('preprocess')
        self.bounding = kwargs.get('bounding')
        if self.bounding is None:
            self.bounding = (0, 1)

    def input_diversity(self, image):
        return image

    def attack(self, image, num_iters):
        if self.random_init:
            self.delta = random_init(image, self.norm_type, self.epsilon)
        else:
            self.delta = torch.zeros_like(image)

        if hasattr(self, 'kernel'):
            self.kernel = self.kernel.to(image.device)

        if hasattr(self, 'grad'):
            self.grad = torch.zeros_like(image)


        epsilon_per_iter = self.epsilon / num_iters * 1.25

        for i in range(num_iters):
            self.delta = self.delta.detach()
            self.delta.requires_grad = True

            image_diversity = self.input_diversity(image + self.delta)
            #plt.imshow(image_diversity.cpu().detach().numpy()[0].transpose(1, 2, 0))
            if self.preprocess is not None:
                image_diversity = self.preprocess(image_diversity)

            yield image_diversity

            grad = self.get_grad()
            grad = self.normalize(grad)
            self.delta = self.delta.data + epsilon_per_iter * grad

            # constraint 1: epsilon
            self.delta = self.project(self.delta, self.epsilon)
            # constraint 2: image range
            self.delta = torch.clamp(image + self.delta, *self.bounding) - image

        yield (image + self.delta).detach()

    def get_grad(self):
        self.grad = self.delta.grad.clone()
        return self.grad

    def project(self, delta, epsilon):
        if self.norm_type == NormType.Linf:
            return torch.clamp(delta, -epsilon, epsilon)
        elif self.norm_type == NormType.L2:
            return clamp_by_l2(delta, epsilon)

    def normalize(self, grad):
        if self.norm_type == NormType.Linf:
            return torch.sign(grad)
        elif self.norm_type == NormType.L2:
            return grad / torch.norm(grad, dim=(1, 2, 3), p=2, keepdim=True)

    def run_trades(self, net, image, num_iters):
        with torch.no_grad():
            origin_output = net.inference_image(self.preprocess(image))
            if self.cls:
                origin_embed = origin_output['image_embed'][:, 0, :].detach()
            else:
                origin_embed = origin_output['image_embed'].flatten(1).detach()

        criterion = torch.nn.KLDivLoss(reduction='batchmean')
        attacker = self.attack(image, num_iters)

        for i in range(num_iters):
            image_adv = next(attacker)
            adv_output = net.inference_image(image_adv)
            if self.cls:
                adv_embed = adv_output['image_embed'][:, 0, :]
            else:
                adv_embed = adv_output['image_embed'].flatten(1)

            loss = criterion(adv_embed.log_softmax(dim=-1), origin_embed.softmax(dim=-1))
            loss.backward()

        image_adv = next(attacker)
        return image_adv

class MirrorDescent:
    def __init__(self, lr) -> None:
        self.model = BertModel.from_pretrained("bert-base-uncased")
        self.lr = lr
        self.eps = 1e-15

    @staticmethod
    def normalize(x, p, eps):
        x_norm = x.view(x.shape[0], -1).norm(p=p, dim=-1)
        return x / (x_norm.view(-1, *((1, ) * (len(x.shape) - 1))) + eps) 
    
    @staticmethod
    def scale(x, eps, dim):
        return x / (torch.sum(x, dim=dim) + eps)
    
    def initialize(self, shape):
        V = self.model.embeddings.word_embeddings.weight
        W = torch.randn(V.shape, requires_grad=True)**2
        W = self.scale(W, self.eps, 0)
        return V, W
    
    def step(self, W):
        # y_{t+1} = x_t exp(- lr * grad f(x_t))
        # x_{t+1} = y_{t+1}  / || y_{t+1} ||_1
        W = W * torch.exp(-self.lr * grad_loss)
        W = self.normalize(W, 1, self.eps)


class ADMM:
    def __init__(
        self,
        device,
        image_attacker,
        text_attacker,
        p = 2,


    ):
        self.device = device
        self.p = p
        self.eps = 1e-15

    @staticmethod
    def normalize(x, p, eps):
        x_norm = x.view(x.shape[0], -1).norm(p=p, dim=-1)
        return x / (x_norm.view(-1, *((1, ) * (len(x.shape) - 1))) + eps) 

    def initialize(self, shape):
        x = torch.randn(shape).to(self.device)
        return self.normalize(x, self.p, self.eps)

    def train():
        self.attack = self.initialize(batch.shape[1:]).unsqueeze(0)





v = v_init
lambd = lambd_init
primal_res = []
dual_res = []
iters = np.arange(0, 500)

cached_inv = np.linalg.inv((Q + tau*np.matmul(A.T, A)))

for k in iters:
    u = np.matmul(cached_inv, np.dot(A.T, lambd+tau*v) - q) # u-update
    v_prev = v
    v = np.minimum(np.dot(A, u) - (lambd/tau), b) # v-update
    lambd = lambd + tau*(v - np.dot(A, u)) # lambda-update

    primal_res.append(np.linalg.norm(v - np.dot(A, u), 2))
    dual_res.append(np.linalg.norm(-tau*np.matmul(A.T, v - v_prev), 2))