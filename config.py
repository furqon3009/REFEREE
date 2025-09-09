def enum(**enums):
    return type('Enum', (), enums)

DatasetsType = enum(
    office31    = 'Office31',
    officehome  = 'OfficeHome',
    mnist       = 'MNIST',
    usps        = 'USPS',
    visda       = 'VisDA',
    domainnet   = 'DomainNet'
)

Office31_src = ['amazon', 'amazon', 'dslr',   'dslr',   'webcam', 'webcam']
Office31_tgt = ['dslr',   'webcam', 'amazon', 'webcam', 'amazon', 'dslr']
OfficeHome_src = ['Art',    'Art',      'Art',          'Clipart',     'Clipart', \
    'Clipart',         'Product',  'Product',  'Product',      'Real World', \
        'Real World',   'Real World']
OfficeHome_tgt = ['Clipart',   'Product',  'Real World',   'Art',      'Product', \
    'Real World',   'Art',      'Clipart',     'Real World',   'Art', \
        'Clipart',         'Product']
DomainNet_src = ['clipart', 'sketch']
DomainNet_tgt = ['sketch', 'clipart']

TaskDetail = {
    'amazon':   [458, 565, 591, 580, 623],
    'dslr':     [97,80,113,89,119],
    'webcam':   [137,152,179,153,174],
    'art':      [257,268,218,122,224,223,234,140,103,153,179,162,144],
    'clip':     [377,395,344,220,339,359,362,338,330,280,301,390,330],
    'product':  [327,327,423,316,392,385,348,370,255,302,333,325,336],
    'world':    [431,398,],
}