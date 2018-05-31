import numpy as np
    
def choose_week(count_matrix,warehouse,random_array,random_index):
    selected_row = count_matrix[warehouse]
    zero_index = np.array(np.nonzero(selected_row==0))
    num = np.size(zero_index)
    if(num != 0):
        addition=random_array[random_index]%num
        random_index += 1
        #zero_index = zero_index.reshape(num,)
        w=zero_index[0,addition]
        return w
    one_index = np.array(np.nonzero(selected_row==1))
    num = np.size(one_index)
    if(num != 0):
           addition=random_array[random_index]%num
           random_index += 1
           #one_index = one_index.reshape(num,)
           w=one_index[0,addition]
           return w
    two_index = np.array(np.nonzero(selected_row==2))
    num = np.size(two_index)
    if(num != 0):
           addition=random_array[random_index]%num
           random_index += 1
           #two_index = two_index.reshape(num,)
           w=two_index[0,addition]
           return w
    three_index = np.array(np.nonzero(selected_row==3))
    num = np.size(three_index)
    if(num != 0):
           addition=random_array[random_index]%num
           random_index += 1
           #three_index = three_index.reshape(num,)
           w=three_index[0,addition]
           return w       
    four_index = np.array(np.nonzero(selected_row==4))
    num = np.size(two_index)
    if(num != 0):
           addition=random_array[random_index]%num
           random_index += 1
           #four_index = four_index.reshape(num,)
           w=four_index[0,addition]
           return w   
    else:
        w = random_array[random_index]%4
        random_index += 1
        return w
def capacity_utilization_pf(a,storage_capacity):
    n = np.ma.size(a)
    b = np.zeros((n))
    for i in range(0,n):
        b[i] = storage_capacity[i]/(a[i])
        b[i] = np.floor(10*b[i])
    return b
	
def surplus_demand(surplus,initial_stock_level,demand,max_allottment,available_space,total_warehouses,total_weeks,storage_capacity,random_array,random_index):
    dv1 = random_array[random_index]
    random_index += 1
    while (surplus>0 and dv1<(total_warehouses*total_weeks/2)):
        max_cupf = capacity_utilization_pf(initial_stock_level+demand,storage_capacity)
        available = np.minimum(max_allottment,available_space) - demand
        max_cupf_ind= max_cupf.argmax()
        max_possible_addition = min(available[max_cupf_ind],surplus)
        addition = (random_array[random_index]%max_possible_addition) + 1
        random_index += 1
        demand[max_cupf_ind] += addition
        surplus -= addition
    while (surplus>0 and dv1>=(total_warehouses*total_weeks/2)):
        available = np.minimum(max_allottment,available_space) - demand
        rand = 0
        selected = 0
        while (selected == 0):
            rand = random_array[random_index]%total_warehouses
            random_index += 1
            selected = available[rand]
        max_possible_addition = min(available[rand],surplus)
        addition = (random_array[random_index]%max_possible_addition) + 1
        random_index += 1
        demand[rand] += addition
        surplus -= addition
    return demand,random_index
	
def surplus_allocation(rake_allocated,demand,total_warehouses,random_array,random_index,count_matrix,key):
    allocated1 = np.sum(rake_allocated)
    allocated = np.sum(allocated1)
    excess = np.array(np.nonzero(demand>4))
    #num = np.size(excess)
    #excess = excess.reshape(num,)
    for i in excess[0]:
        dem = demand[i]
        while(dem > 8-demand[i]):
            if(key==1):
                w = random_array[random_index]%4
                random_index += 1
            else:
                w=choose_week(count_matrix,i,random_array,random_index)
            if(rake_allocated[i][w]==0) :
                rake_allocated[i][w] += 2
                count_matrix[i][w] += 2
                allocated += 2
                dem -= 2
    return allocated,rake_allocated,random_index,count_matrix
	

def rake_allocation(allocated,total_rakes,rake_allocated,monthly_allotment,comb_matrix,comb,k_shift,k_reset,k_terminate,terminal_capacity,total_warehouses,total_weeks,random_array,random_index,count_matrix,initial_count_matrix,key):
    k = 1
    while allocated<total_rakes:
        j = random_array[random_index]%total_warehouses
        random_index += 1
        if(key==1):
            w = random_array[random_index]%total_weeks
            random_index += 1
        else:
            w=choose_week(count_matrix,j,random_array,random_index)
             
        
        #selected_row = rake_allocated[j]
        sum4 = sum(rake_allocated[j])
        dv2 = random_array[random_index]
        random_index += 1
        partition = 500
        if ((sum4<monthly_allotment[j]) and (rake_allocated[j][w]==0)):
                #comb_row = comb_matrix[j]
                combwh = np.array(np.nonzero(comb_matrix[j]==1))
                #num = np.size(combwh)
                #combwh = combwh.reshape(num,)
                sum_ra = np.sum(rake_allocated,axis=1)
                remaining_aloc=monthly_allotment-sum_ra
                c=np.array(np.nonzero(remaining_aloc[combwh[0]]>0))
                possible_wh=combwh[0,c]
                num = np.size(possible_wh)
                #possible_wh = possible_wh.reshape(num)
                if(num>0):
                    minimum=possible_wh[0,0]
                    temp=1
                    while(temp<num):
                      if(count_matrix[possible_wh[0,temp]][w]<count_matrix[minimum][w]):
                        minimum = possible_wh[0,temp]
                      temp+=1
                    cw = minimum
                    #selected_row2 = rake_allocated[cw,:]
                    #sum6 = sum(selected_row2)
                    sum6 = sum_ra[cw]
                    while(rake_allocated[cw][w]==0 and sum6<monthly_allotment[cw] and dv2<partition):
                        rake_allocated[j][w] += 1
                        count_matrix[j][w] += 1
                        rake_allocated[cw][w] += 1
                        count_matrix[cw][w] += 1
                        comb[j][cw][w] += 1
                        comb[cw][j][w] += 1
                        allocated += 2
                        sum4 += 1
                        sum6 += 1
                    if(dv2>=partition and k>k_shift and (sum4+1)<monthly_allotment[j] and (terminal_capacity[j]==2 and rake_allocated[j][w]==0)):
                            if(sum4<(monthly_allotment[j]-1)):
                                rake_allocated[j][w] += 2
                                count_matrix[j][w] += 2
                                allocated += 2
                                sum4 += 2
                elif(num == 0 and (sum4+1)<monthly_allotment[j]):
                    rake_allocated[j][w] += 2
                    count_matrix[j][w] += 2
                    allocated += 2
                    sum4 += 2
            
        if(k%k_reset == 0):
            rake_allocated = np.zeros((total_warehouses,total_weeks))
            count_matrix=initial_count_matrix.copy()
            comb = np.zeros((total_warehouses,total_warehouses,4))
            allocated,rake_allocated,random_index,count_matrix = surplus_allocation(rake_allocated,monthly_allotment,total_warehouses,random_array,random_index,count_matrix,key)
        if(k>k_terminate):
            break
        k += 1
    return rake_allocated,comb,allocated,random_index,count_matrix
	
def penalty(rake_allocated,rake_penalty_h,rake_penalty_f,initial_stock_level,weekly_penalty,count_matrix):
    half_rakes = np.array(np.nonzero(rake_allocated==1))
    full_rakes = np.array(np.nonzero(rake_allocated==2))
    size1 = np.size(half_rakes)
    size2 = np.size(full_rakes)
    cost1 = size1*rake_penalty_h/2
    cost2 = size2*rake_penalty_f/2
    rake_penalty = cost1+cost2

    alloted_weeks = np.array(np.nonzero(rake_allocated>0))
    week_penalty = 0
    weeknum = np.size(alloted_weeks)/2
    m = 0
    while m < weeknum:
        swh = alloted_weeks[0,m]
        sw = alloted_weeks[1,m]
        week_penalty += weekly_penalty[swh][sw]
        m += 1

#    capacity_utilization_penalty = np.sum(capacity_utilization_pf(initial_stock_level+demand,storage_capacity))
    total_penalty = rake_penalty + week_penalty
    return total_penalty
	
import time

def simulate(iteration,total_warehouses,total_weeks,total_rakes,demand,initial_stock_level,storage_capacity,terminal_capacity,max_allottment,rake_penalty_h,rake_penalty_f,weekly_penalty,k_shift,k_reset,k_terminate,count_matrix,comb_matrix,key):
    weekly_penalty = weekly_penalty.reshape(total_warehouses,total_weeks)
    comb_matrix = comb_matrix.reshape(total_warehouses,total_warehouses)
    random_array = np.random.randint(0,total_warehouses*total_weeks,50000*iteration)
    random_index = 0
    first = 1
    original_demand = demand.copy()
    initial_count_matrix=count_matrix.copy()
    for i in range(0,iteration):
        count_matrix=initial_count_matrix.copy()
        demand = original_demand.copy()
        rake_allocated = np.zeros((total_warehouses,4))
        total_demand = np.sum(demand)
        surplus = total_rakes - total_demand
        available_space = storage_capacity - initial_stock_level
        demand,random_index = surplus_demand(surplus,initial_stock_level,demand,max_allottment,available_space,total_warehouses,total_weeks,storage_capacity,random_array,random_index)
        comb = np.zeros((total_warehouses,total_warehouses,total_weeks))
        monthly_allotment = demand
        allocated,rake_allocated,random_index,count_matrix = surplus_allocation(rake_allocated,demand,total_warehouses,random_array,random_index,count_matrix,key)
        rake_allocated,comb,allocated,random_index,count_matrix = rake_allocation(allocated,total_rakes,rake_allocated,monthly_allotment,comb_matrix,comb,k_shift,k_reset,k_terminate,terminal_capacity,total_warehouses,total_weeks,random_array,random_index,count_matrix,initial_count_matrix,key)
        
        total_penalty = penalty(rake_allocated,rake_penalty_h,rake_penalty_f,initial_stock_level,weekly_penalty,count_matrix)
        if(first ==1 and allocated == total_rakes):
            final_weekly_distribution = rake_allocated
            final_count_matrix=count_matrix
            final_total_penalty = total_penalty
            final_comb = comb
            final_monthly_allotment=monthly_allotment
            first += 1
        if(allocated == total_rakes and first != 1):
            if(total_penalty<final_total_penalty):
                final_weekly_distribution = rake_allocated
                final_total_penalty = total_penalty
                final_comb = comb
                final_monthly_allotment=monthly_allotment
                final_count_matrix=count_matrix
        if(random_index>=(0.9*50000*iteration)):
           random_array = np.random.randint(0,total_warehouses*total_weeks,50000*iteration)
           random_index = 0    
   
    print(final_weekly_distribution)
    print(final_total_penalty)
    return final_count_matrix,final_monthly_allotment,final_total_penalty,final_weekly_distribution
def main():
    start = time.time()
    total_warehouses = 14
    total_weeks = 4
    total_rakes_b = 36
    total_rakes_r = 26
    total_rakes_w = 18
    demand_b = np.array([4,1,1,4,4,2,4,3,4,3,1,2,2,1])
    demand_r = np.array([3,0,1,5,5,2,1,2,3,2,0,1,1,0])
    demand_w = np.array([2,2,0,2,3,1,2,1,2,1,0,1,0,1])
    initial_stock_level_rice = np.array([22,18,6,10,23,26,39,5,10,2,17,7,14,2])
    initial_stock_level_wh = np.array([4,6,2,3,6,10,9,3,4,2,6,2,5,1])
    total_initial_stock_level=initial_stock_level_rice+initial_stock_level_wh
    storage_capacity = np.array([28,23,7,15,31,38,56,7,27,3,35,8,23,7])
    terminal_capacity = np.array([2,2,1,2,2,2,2,1,1,1,2,2,2,1])           
    max_allottment = np.array([8,8,8,8,8,8,8,8,8,8,8,8,8,8])
    rake_penalty_h =20;
    rake_penalty_f = 50;
    weekly_penalty_rice =    np.array([10,1,1,1,10,1,1,1,10,1,1,2,10,1,1,1,2,2,1,1,10,1,1,1,3,1,1,1,5,2,1,4,5,2,1,1,2,2,1,1,10,1,1,4,4,2,1,2,4,1,1,2,3,2,1,1])
    weekly_penalty_wh =    np.array([10,2,1,1,10,2,1,1,10,1,1,1,3,2,1,2,1,1,1,1,10,1,1,1,4,1,2,1,10,1,1,2,7,3,1,2,10,2,1,1,10,1,1,1,1,2,2,1,10,1,1,1,10,1,1,1])
    k_shift = 135
    k_reset = 575
    k_terminate = 7450
    comb_matrix =    np.array([0,1,1,1,0,0,0,0,0,0,0,0,0,0,1,0,1,1,1,1,0,0,0,0,0,0,0,0,1,1,0,1,1,1,0,1,0,1,0,0,0,0,1,1,1,0,1,1,0,0,0,0,0,0,0,0,0,1,1,1,0,1,1,1,1,1,1,0,0,0,0,1,1,1,1,0,1,1,1,1,1,1,0,0,0,0,0,0,1,1,0,1,1,1,1,0,0,0,0,0,1,0,1,1,1,0,1,1,1,1,0,0,0,0,0,0,1,1,1,1,0,1,1,1,1,1,0,0,1,0,1,1,1,1,1,0,1,1,1,1,0,0,0,0,1,1,1,1,1,1,0,1,1,1,0,0,0,0,0,1,0,1,1,1,1,0,1,1,0,0,0,0,0,0,0,0,1,1,1,1,0,1,0,0,0,0,0,0,0,0,1,1,1,1,1,0])
    iteration=5000
    count_matrix=np.zeros((total_warehouses,4))
    print("Boiled rice")
    count_matrix,new_stock_b,penalty_b,weekly_distribution_b=simulate(iteration,total_warehouses,total_weeks,total_rakes_b,demand_b,initial_stock_level_rice,storage_capacity,terminal_capacity,max_allottment,rake_penalty_h,rake_penalty_f,weekly_penalty_rice,k_shift,k_reset,k_terminate,count_matrix,comb_matrix,1)
    print("Raw rice")
    initial_stock_level_rice=initial_stock_level_rice+new_stock_b
    count_matrix,new_stock_r,penalty_r,weekly_distribution_r=simulate(iteration,total_warehouses,total_weeks,total_rakes_r,demand_r,initial_stock_level_rice,storage_capacity,terminal_capacity,max_allottment,rake_penalty_h,rake_penalty_f,weekly_penalty_rice,k_shift,k_reset,k_terminate,count_matrix,comb_matrix,2)
    initial_stock_level_rice=initial_stock_level_rice+new_stock_r
    initial_stock_level_wh= initial_stock_level_wh+initial_stock_level_rice
    print("Wheat") 
    count_matrix,new_stock_w,penalty_w,weekly_distribution_w=simulate(iteration,total_warehouses,total_weeks,total_rakes_w,demand_w,initial_stock_level_wh,storage_capacity,terminal_capacity,max_allottment,rake_penalty_h,rake_penalty_f,weekly_penalty_wh,k_shift,k_reset,k_terminate,count_matrix,comb_matrix,3)
    print(count_matrix)
    penalty_sum=penalty_r+penalty_b+penalty_w
    total_new_stock=new_stock_b+new_stock_r+new_stock_w
    capacity_utilization_penalty = np.sum(capacity_utilization_pf(total_initial_stock_level+total_new_stock,storage_capacity))
    penalty_sum=penalty_sum+capacity_utilization_penalty 
    print(penalty_sum)
    end = time.time()
    print(end-start)
main()    