import numpy as np
import matplotlib.pyplot as plt
import time
# Parameters
delta_t = 1.
epsilon = 1e-16 # noise
alpha = 0.04
k = 1.
eps_sqrt = np.sqrt(epsilon)
steps = int(1e4)
n_clones = int(1e3) # Max number of clones

def eq_p(q,p,noise):
    return p - k*delta_t*np.sin(2.*np.pi*q)/(2.*np.pi) + eps_sqrt*noise
def eq_q(q,p):
    return q + delta_t*p
def module(v1,v2):
    return np.sqrt(v1**2+v2**2)
def boundary(q,p):
    for l in range(len(q)):
        q[l] = q[l]%1.
        if (q[l]<=0):
            q[l] = q[l] +1
        if (p[l]>=-0.5):
            p[l] = (0.5+p[l]) % 1.-0.5
        else:
            p[l] = 0.5-(0.5+(p[l]) % 1.)
    return q,p

def LWD(delta_t, epsilon, steps, n_clones, q, p, func_q, func_p):
    t1 = time.time()
    scale= 1e-6
    q_traj = np.ones(0)
    p_traj = np.ones(0)
    u_q = np.random.uniform(0,1,n_clones)
    u_p = np.random.uniform(0,1,n_clones)
    u_mods = np.zeros((steps,n_clones))
    
    indices = np.arange(0,n_clones)
    pop_counter = np.zeros(steps,dtype=int)
    pop_counter[0] = n_clones
    
    # Normalization
    u_norm = module(u_q,u_p)
    u_p = scale*u_p/u_norm
    u_q = scale*u_q/u_norm
    u_mods[0][:] = scale

    randomstate = np.random.default_rng()
    for i in range(1,steps):
        # Generate all necessary random numbers
        random_noise = np.random.normal(loc = 0.0, scale = 1.0, size=n_clones) 
        random_selection = np.random.uniform(size=n_clones) 

        p = func_p(q,p,random_noise)
        q = func_q(q,p)
        
        ## Non differentiable
        p_c = func_p(q+u_q,p+u_p,random_noise)
        q_c = func_q(q+u_q,p_c)

        u_p = p_c-p
        u_q = q_c-q
        
        ## Differentiable
        # u_p = -A_pj vj (with auxiliary)
        # u_q = -A_qj vj

        ## Saving modules
        u_mods[i][:] = module(u_q,u_p)
        
        ## Scaling
        u_q = scale*u_q/u_mods[i,:]
        u_p = scale*u_p/u_mods[i,:]


        ## Calculating tau
        pop_loc_counter = np.floor(random_selection+(u_mods[i][:]/scale)**alpha).astype(int)
        pop_counter[i] = np.sum(pop_loc_counter).astype(int)

        ## Asigning populations
        if (pop_counter[i] == 0):
            print("Population has crashed (0) at t: ", i)
            break
        if (pop_counter[i] > n_clones):
            # generating the total population array
            all_pop = np.cumsum(np.append([0],pop_loc_counter))
            # Selecting n_clones of them without replacement
            selected = randomstate.choice(pop_counter[i], size = n_clones, replace=False)
            # Seeing how many fall on which box
            howmany,b = np.histogram(selected,bins = all_pop)
            # Assigning the number of found repetitions
            q = np.repeat(q,howmany)
            p = np.repeat(p,howmany)
        elif (pop_counter[i] < n_clones):
            ## Filling up missing population
            new_indices = randomstate.choice(indices,
                                             size = n_clones-pop_counter[i],
                                             replace=True,
                                             p = pop_loc_counter/pop_counter[i])
            q_short = np.repeat(q,pop_loc_counter)
            p_short = np.repeat(p,pop_loc_counter)
            q = np.append(q_short,q[new_indices])
            p = np.append(p_short,p[new_indices])
        else:
            ## Generating the population
            q = np.repeat(q,pop_loc_counter)
            p = np.repeat(p,pop_loc_counter)
        if (i % int(steps/10)*10==0):
            print(str(i/(steps/10)*10) + "%")
        # To check
        #q,p = boundary(q,p)

        n_sample = 40
        min_track = steps -5000
        if (i>min_track):
            q_aux = q[0:n_sample]
            p_aux = p[0:n_sample]

            q_aux,p_aux = boundary(q_aux,p_aux)
            q_traj = np.append(q_traj,q_aux)
            p_traj = np.append(p_traj,p_aux)
    print("Elapsed time: ", time.time()-t1)
    return q_traj,p_traj, pop_counter, u_mods

def trajectory_plot(ax, tray_steps:int =1000, tray_clones:int = 1000):
    print("Drawing trajectories...")
    tray = np.zeros((tray_steps,2,tray_clones))
    q = np.random.uniform(0,1,tray_clones)
    p = np.random.uniform(-0.5,0.5,tray_clones)
    for i in range(tray_steps):
        p = eq_p(q,p,0)
        q = eq_q(q,p)
        tray[i,0,:],tray[i,1,:] = q,p
    for i in range(tray_clones):
        ax.plot(tray[:-1,0,i],tray[:-1,1,i],"r-",linewidth=1./tray_clones)
    return ax
def mu_calc(pop_counter,ax):
    R = np.zeros(steps-1)
    for i in range(steps-1):
        R[i] = pop_counter[i+1]/pop_counter[i]
    log_R = np.log(R)
    Z = np.prod(log_R)
    mu = np.sum(log_R)/steps
    ax.plot(np.arange(0,pop_counter.size),pop_counter/n_clones)
    return mu,ax

q = np.ones(n_clones)*0.5
p = np.zeros(n_clones)
#q = np.ones(n_clones)*0.207+np.random.uniform()*0.05-0.025
#p = np.ones(n_clones)*0.09+np.random.uniform()*0.05-0.025

q,p, pop_counter, u_mods = LWD(delta_t,epsilon,steps,n_clones,q,p,eq_q,eq_p)
fig,ax = plt.subplots()
mu,ax = mu_calc(pop_counter,ax)
fig.savefig("mu.png")

fig,ax =plt.subplots()
trajectory_plot(ax)
ax.scatter(q,p,s=.05,alpha=0.1)
ax.set_xlim([0,1])
ax.set_ylim([-0.5,0.5])
fig.savefig("trajectories.png")

print(mu)
