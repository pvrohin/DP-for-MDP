import numpy as np
import matplotlib.pyplot as plt
from timeit import default_timer as timer





print("Initialising world, policies, model-environment....")

World = np.array([[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1], 
[1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
[1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
[1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
[1,0,0,0,1,1,1,1,1,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,1,1,1,1,1,0,0,0,1],
[1,1,1,1,1,1,1,1,1,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,1,1,1,1,1,0,0,0,1],
[1,1,1,1,1,1,1,1,1,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,1,1,1,1,1,0,0,0,1],
[1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1],
[1,1,1,1,1,1,1,1,1,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,1,1,1,1,1,1,1,1,1],
[1,1,1,1,1,1,1,1,1,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,1,1,1,1,1,1,1,1,1],
[1,0,0,0,1,1,1,1,1,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,1,1,1,1,1,0,0,0,1],
[1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
[1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]], np.int32)


states = np.zeros((14,51,2)) # states will be a 2d array, where each element is a vector of size 2, containing the row number and column number 


# actions = np.array([[-1,1],[0,1],[1,1],[-1,0],[1,0],[-1,-1],[0,-1],[1,-1]]) # Not used, for the deterministic case.

# 0->up, 1->down, 2->left, 3->right, 4->north west, 5-> north east, 6-> south west, 7-> south east

rewards = np.zeros((14,51))

for i in range(World.shape[0]):
	for j in range(World.shape[1]):

		states[i][j][0] = i
		states[i][j][1] = j
		if World[i][j]==1:
			rewards[i][j] = -50
		else:
			rewards[i][j] = -1



rewards[7][10] = 100

gamma = 0.95
delta = 0.1

states = np.reshape(states,(14*51,2)) # flattening for ease of parsing
V = [0 for s in states] # V is a 1-D array of size 14*51



for s in states: # We want to exclude those states, where there are obstacles, that is, we cannot be there. (-100000 is a randomly chosen identifier)
	if World[int(s[0])][int(s[1])] == 1:
		V[int(s[0])*51 + int(s[1])] = -100000


# Set policy iteration parameters
max_policy_iter = 10000  # Maximum number of policy iterations
max_value_iter = 10000  # Maximum number of value iterations



def actions_possible(state): #Given a state, it calculates all the 8 neighbouring states ( if it is at the edges, special precuations are taken (not necessary!!)) 
# In reality these are actions you are taking from a state
	i = state[0]
	j = state[1]

	if (i>0 and j>0 and i< 13 and j<50):
		actions = [[i-1,j-1],[i-1,j],[i-1,j+1],[i,j-1],[i,j+1],[i+1,j-1],[i+1,j],[i+1,j+1]] # [up left, up, up right, left,right,down left,down, down right]
		#[[i-1,j+1],[i,j+1],[i+1,j+1],[i-1,j],[i+1,j],[i-1,j-1],[i,j-1],[i+1,j-1]] # [up right, right, down right, up,down,up left,left, down left]

	elif(i == 0):
		if(j>0 and j < 50):
			actions = [[i,j+1],[i+1,j+1],[i+1,j],[i,j-1],[i+1,j-1]]
		elif(j == 0):
			actions = [[i,j+1],[i+1,j+1],[i+1,j]]
		elif(j == 50):
			actions = [[i+1,j],[i,j-1],[i+1,j-1]]

	elif(i == 13):
		if(j>0 and j < 50):
			actions = [[i-1,j+1],[i,j+1],[i-1,j],[i-1,j-1],[i,j-1]]
		elif(j == 0):
			actions = [[i-1,j+1],[i,j+1],[i-1,j]]
		elif(j == 50):
			actions = [[i-1,j],[i-1,j-1],[i,j-1]]

	elif(j==0):
		if(i>0 and i<13):
			actions = [[i-1,j+1],[i,j+1],[i+1,j+1],[i-1,j],[i+1,j]]

	elif(j ==50):
		if(i>0 and i<13):
			actions = [[i-1,j],[i+1,j],[i-1,j-1],[i,j-1],[i+1,j-1]]

	return actions



def policy_update(policy,state,optimal_state,close_states): # If a better policy (that is action for a given state) is found, then the policy is updated to take that action at that state

	for next_state in close_states:
		policy[str(state[0]*51+ state[1])+'|'+str(next_state[0]*51+next_state[1])] = 0.0

	policy[str(state[0]*51+ state[1])+'|'+str(optimal_state[0]*51+optimal_state[1])] = 1.0

	return policy



def model_distribution(from_state,to_state):
	model_probability = np.zeros(8) # [up left, up, up right, left,right,down left,down, down right]
	direction = to_state-from_state

	action = direction[0]*3 + direction[1] + 4

	if action>4:
		action += -1

	model_probability[int(action)] = 0.6 # If deterministic is required keep this as 1


	if direction[1] == 0:
		action_45_left = (direction[0])*3 + direction[1]-1+4
		action_45_right = (direction[0])*3 + direction[1]+1+4

	elif direction[0] == 0:
		action_45_left = (direction[0]-1)*3 + direction[1]+4
		action_45_right = (direction[0]+1)*3 + direction[1]+4

	elif direction[0] == -1 and direction[1] == -1:
		action_45_left = (direction[0]+1)*3 + direction[1]+4
		action_45_right = direction[0]*3 + direction[1]+1+4

	elif direction[0] == -1 and direction[1] == 1:
		action_45_left = (direction[0])*3 + direction[1]-1+4
		action_45_right = (direction[0]+1)*3 + direction[1]+4

	elif direction[0] == 1 and direction[1] == 1:
		action_45_left = (direction[0]-1)*3 + direction[1]+4
		action_45_right = (direction[0])*3 + direction[1]-1+4

	elif direction[0] == 1 and direction[1] == -1:
		action_45_left = (direction[0]-1)*3 + direction[1]+4
		action_45_right = (direction[0])*3 + direction[1]+1+4

	if action_45_right >4:
		action_45_right += -1

	if action_45_left >4:
		action_45_left += -1

	model_probability[int(action_45_left)] = 0.2 # If deterministic is required keep this as 0
	model_probability[int(action_45_right)] = 0.2 # If deterministic is required keep this as 0

	return model_probability


 
pi = {} #creating random policy. Initially from any state, the probability of going to any of the directions possible (provided that direction is allowable) is same.


for s in states:

	actions = actions_possible([s[0],s[1]])
	# pi_state = np.zeros(len(close_states))
	pi_state = 1/len(actions)

	for action in actions:

		pi[str(s[0]*51+s[1])+'|'+str(action[0]*51+action[1])] = pi_state # It is opposite notation, that is this is pi(taking action a which leads to position state[0][1] given you are in state s[0][1])
																	   # Note that since we are only looking at close states which are accessible through the possible set of actions in a deterministic environment
																	   # going to a close state and taking an action which allows us to go to the close state is the same.


print("Starting timer for value iteration till optimal policy is found")
start = timer() # iterative loops start to find the optimal policy
for i in range(max_policy_iter):
	# Initial assumption: policy is stable
	optimal_policy_found = True
	# Policy evaluation
	# Compute value for each state under current policy

# Value iteration
	for j in range(max_value_iter):
		max_diff = 0  # Initialize max difference
		
		for s in states:

			if  V[int(s[0])*51 + int(s[1])] != -100000: # Neglecting those states where you are in obstacles

		#Compute state value

				v_new = 0

				actions = actions_possible([s[0],s[1]])
				possible_states = actions
				
				for action in actions:

					prob_dist = model_distribution(s,action)

					for arg, prob in enumerate(prob_dist):

						if V[int(possible_states[arg][0])*51 + int(possible_states[arg][1])] == -100000: # If you choose an action which goes into the obstacle, you come back to the same state
							V_action = V[int(s[0])*51 + int(s[1])]
						else:
							V_action = V[int(possible_states[arg][0])*51 + int(possible_states[arg][1])]

						v_new += pi[str(s[0]*51+s[1]) + '|' + str(action[0]*51+action[1])]*prob*(rewards[int(possible_states[arg][0])][int(possible_states[arg][1])] + gamma * V_action) 



				# Update maximum difference

				max_diff = max(max_diff, abs(v_new - V[int(s[0])*51 + int(s[1])]))

				V[int(s[0])*51 + int(s[1])] = v_new  # Update value with highest value
			# If diff smaller than threshold delta for all states, algorithm terminates

		
		if max_diff < delta:
			break

# Policy iteration
	for s in states:

		if  V[int(s[0])*51 + int(s[1])] != -100000: # Neglecting those states where you are in obstacles

			actions = actions_possible([s[0],s[1]])
			possible_states = actions

			v_curr =  V[int(s[0])*51 + int(s[1])] #Current value based on the current policy

			val_max = -100000
			best_action = actions[0]

			for action in actions:

				val = 0

				prob_dist = model_distribution(s,action)

				for arg, prob in enumerate(prob_dist):

					if V[int(possible_states[arg][0])*51 + int(possible_states[arg][1])] == -100000: # If you choose an action which goes into the obstacle, you come back to the same state
						V_action = V[int(s[0])*51 + int(s[1])]
					else:
						V_action = V[int(possible_states[arg][0])*51 + int(possible_states[arg][1])]

					val += prob*(rewards[int(possible_states[arg][0])][int(possible_states[arg][1])] + gamma * V_action) 


				if(val>val_max):
					val_max = val
					best_action = action

				# elif(val==val_max): ## essentially same thing, but more aesthatically pleasing
				# 	direct = action-s
				# 	if(direct[0]==0 or direct[1]== 0):
				# 		val_max = val
				# 		best_action = action



		

			if (pi[str(s[0]*51+s[1]) + '|' + str(best_action[0]*51+best_action[1])] != 1): # If a different action led to a better action value, then update policy and 
																				 # then we will update values as per previous algo
				optimal_policy_found = False


				pi = policy_update(pi,s,best_action,actions)
						


	# If policy did not change, algorithm terminates
	if optimal_policy_found:
		break


end = timer() # Optimal policy found
print("Optimal policy found")
print("Time taken to find the optimal policy is: ", end - start)


# Removing the randomly chosen identifier and setting the value function there to zero.
for s in states:
	if V[int(s[0])*51 + int(s[1])] == -100000:
		V[int(s[0])*51 + int(s[1])] = 0


V = np.reshape(V,(14,51))



fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

# Major ticks every 20, minor ticks every 5
major_ticks_x = np.arange(-0.5, 51, 1)
major_ticks_y = np.arange(-0.5,14,1)
# minor_ticks = np.arange(0, 101, 5)

ax.set_xticks(major_ticks_x)
# ax.set_xticks(minor_ticks, minor=True)
ax.set_yticks(major_ticks_y)
#ax.set_yticks(minor_ticks, minor=True)

# And a corresponding grid
ax.grid(which='both',color='k')

# Or if you want different settings for the grids:
ax.grid(which='major', alpha=0.2)

#plt.show()

plt.imshow(V,cmap='gray')
plt.show()

 
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

ax.set_xticks(major_ticks_x)
# ax.set_xticks(minor_ticks, minor=True)
ax.set_yticks(major_ticks_y)
#ax.set_yticks(minor_ticks, minor=True)

# And a corresponding grid
ax.grid(which='both')

# Or if you want different settings for the grids:
ax.grid(which='major', alpha=0.2,color='k')

def optimal_action(pi,state):
	actions = actions_possible([state[0],state[1]])

	best_action = actions[0]

	for action in actions:
		if (pi[str(state[0]*51+state[1]) + '|' + str(action[0]*51+action[1])] > 0.9):
			best_action = action
			break

	direction = best_action - state

	length = np.sqrt(direction[0]*direction[0]+direction[1]*direction[1])
	direction = direction*0.5/length

	return direction

for s in states:
	if int(V[int(s[0])][int(s[1])]) != 0:

		direction = optimal_action(pi,s)
		plt.arrow(s[1],s[0],direction[1],direction[0],head_width=0.3)

#direction = optimal_action(pi,[13,12])
direction = [0,-1]


[x,y] = np.shape(World)

output = list(np.zeros(np.shape(World)))

for i in range(x):
	for j in range(y):
		if (World[i][j] == 0):
			output[i][j] = 1


plt.imshow(output,cmap='gray')
plt.show()

