print('Importing libraries...')
import pickle
import torch
import chipmunk
chipmunk.util.config.load_from_file('chipmunk-config.yml')

print('Loading tensors...')
q,k,v,inds,counts = pickle.load(open('saved_tensors.pkl', 'rb'))

print('q shape:', q.shape)
print('k shape:', k.shape)
print('v shape:', v.shape)
print('inds shape:', inds.shape)
print('counts shape:', counts.shape)

print('Checking indices...')

# for b in range(inds.shape[0]):
#     for h in range(inds.shape[1]):
#         for group_idx in range(inds.shape[2]):
#             limit = 76800+256*92
#             counts[b,h,group_idx] = min(counts[b,h,group_idx].item(), limit)
#             inds[b,h,group_idx,:counts[b,h,group_idx]] = torch.arange(q.shape[-2])[:counts[b,h,group_idx]]
#             inds[b,h,group_idx,:] = torch.arange(q.shape[-2])[:]

#             count = counts[b,h,group_idx]
#             inds_b = inds[b,h,group_idx,:count]
#             test_tensor = torch.all((inds_b >= 0) & (inds_b < q.shape[-2]))
#             # inds[b,h,group_idx,:count] = torch.minimum(torch.maximum(inds_b, torch.tensor(0)), torch.tensor(limit-1))
#             assert test_tensor.item()
# o = chipmunk.ops.dense_attn(q, k, v)
# o = chipmunk.ops.csp_attn(q, k, v, inds, counts)

# chipmunk.util.config.GLOBAL_CONFIG['attn']['provider'] = 'cuda'
# o_ref = chipmunk.ops.csp_attn(q, k, v, inds, counts)

# torch.cuda.synchronize()
# print(o)

# breakpoint()