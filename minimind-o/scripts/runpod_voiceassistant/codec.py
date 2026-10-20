"""VoiceAssistant SNAC frame order -> three temporal codec levels."""
def split_snac(value):
 groups=[s.strip().split() for s in value.split('#') if s.strip()]
 if not groups or any(len(g)!=7 for g in groups):raise ValueError('Expected groups of seven SNAC tokens')
 levels=[[],[],[]]
 for group in groups:
  g=[int(x) for x in group]
  if any(x<0 or x>=4096 for x in g):raise ValueError('SNAC token outside [0,4096)')
  levels[0].append(g[0]);levels[1].extend([g[1],g[4]]);levels[2].extend([g[2],g[3],g[5],g[6]])
 return levels
