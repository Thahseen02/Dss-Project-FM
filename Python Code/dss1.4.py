import numpy as np
##penalty check with capacity is done
##selecting week by considering only number of rakes allocated
##H+H is done
def choose_week(count_matrix,warehouse,random_array,random_index):
    selected_row = count_matrix[warehouse]
    zero_index = np.array(np.nonzero(selected_row==0))
    num = np.size(zero_index)
    if(num != 0):
        addition=random_array[random_index]%num
        random_index += 1
        w=zero_index[0,addition]
        return w
    one_index = np.array(np.nonzero(selected_row==1))
    num = np.size(one_index)
    if(num != 0):
           addition=random_array[random_index]%num
           random_index += 1
           w=one_index[0,addition]
           return w
    two_index = np.array(np.nonzero(selected_row==2))
    num = np.size(two_index)
    if(num != 0):
           addition=random_array[random_index]%num
           random_index += 1
           w=two_index[0,addition]
           return w
    three_index = np.array(np.nonzero(selected_row==3))
    num = np.size(three_index)
    if(num != 0):
           addition=random_array[random_index]%num
           random_index += 1
           w=three_index[0,addition]
           return w       
    four_index = np.array(np.nonzero(selected_row==4))
    num = np.size(two_index)
    if(num != 0):
           addition=random_array[random_index]%num
           random_index += 1
           w=four_index[0,addition]
           return w   
    else:
        w = random_array[random_index]%4
        random_index += 1
        return w
    

def full_rake_split(warehouse,week,rake_allocated,count_matrix,comb_matrix,comb,total_weeks,total_warehouses,weekly_penalty,monthly_allotment,allocated,dem):
    combwh = np.array(np.nonzero(comb_matrix[warehouse]==1))   ##combination warehouses
    sum_ra = np.sum(rake_allocated,axis=1)
    remaining_aloc=monthly_allotment-sum_ra
    c=np.array(np.nonzero(remaining_aloc[combwh[0]]>0))    
    possible_wh=combwh[0,c]                                     ##warehouses whose demand where not met yet
    num = np.size(possible_wh)
    firstmin=-1
    secondmin=-1
    if(num>=2):
        temp=0
        while(temp<num):
            if(rake_allocated[possible_wh[0,temp]][week]==0):
                if(weekly_penalty[possible_wh[0,temp]][week]<weekly_penalty[firstmin][week]):
                        firstmin = possible_wh[0,temp]
                            
            temp+=1
        temp=0
        while(temp<num):
            if(rake_allocated[possible_wh[0,temp]][week]==0):
                if((weekly_penalty[possible_wh[0,temp]][week]<weekly_penalty[secondmin][week]) and(possible_wh[0,temp] != firstmin)):
                        secondmin = possible_wh[0,temp]
                            
            temp+=1    

    if((firstmin!=-1) and (secondmin!=-1)):                    ###H+H is possible
             rake_allocated[warehouse][week]= 2                    ###H+H is indicated by 2 
             count_matrix[warehouse][week]+= 2
             rake_allocated[firstmin][week]= 1
             count_matrix[firstmin][week] += 1
             rake_allocated[secondmin][week]= 1
             count_matrix[secondmin][week] += 1
             comb[warehouse][firstmin][week] = 1
             comb[warehouse][secondmin][week] = 1
             comb[secondmin][warehouse][week] = 1
             comb[firstmin][warehouse][week] = 1
             allocated += 4
             dem -= 4    
             
    else:                    ##only full rake is possible,no H+H
             rake_allocated[warehouse][week] = 3
             count_matrix[warehouse][week] += 2
             allocated += 2
             dem -= 2
    return count_matrix,rake_allocated,comb,allocated,dem 


def find_sum_col(weekly_distribution):
    full_count=np.count_nonzero(weekly_distribution==2,axis=0)
    rakes_sum=np.sum(weekly_distribution,axis=0)
    total_rakes_sum_col=rakes_sum-full_count
    return total_rakes_sum_col

def find_row_sum(a,b,c):
    row_wise_sum=a+b+c
    row_wise_sum=np.sum(row_wise_sum,axis=1)
    full_count_a=np.count_nonzero(a==2,axis=1)
    full_count_b=np.count_nonzero(b==2,axis=1)
    full_count_c=np.count_nonzero(c==2,axis=1)
    total_full_count=full_count_a+full_count_b+full_count_c
    total_row_wise_sum=row_wise_sum-total_full_count
    return total_row_wise_sum


    
    
    
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
	
def surplus_allocation(rake_allocated,demand,total_warehouses,random_array,random_index,count_matrix,key,weekly_penalty,comb_matrix,comb,total_weeks):
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
#                rake_allocated[i][w] = 2
#                count_matrix[i][w] += 2
               count_matrix,rake_allocated,comb,allocated,dem=full_rake_split(i,w,rake_allocated,count_matrix,comb_matrix,comb,total_weeks,total_warehouses,weekly_penalty,demand,allocated,dem)
    return allocated,rake_allocated,random_index,count_matrix
	

def rake_allocation(allocated,total_rakes,rake_allocated,monthly_allotment,comb_matrix,comb,k_shift,k_reset,k_terminate,terminal_capacity,total_warehouses,total_weeks,random_array,random_index,count_matrix,initial_count_matrix,key,weekly_penalty):
    k = 1
    while allocated<total_rakes:
        j = random_array[random_index]%total_warehouses
        random_index += 1
        if(key==1):
            w = random_array[random_index]%total_weeks
            random_index += 1
        else:
            w=choose_week(count_matrix,j,random_array,random_index)
        sum4 = sum(rake_allocated[j])
        dv2 = random_array[random_index]
        random_index += 1
        partition = 500
        if ((sum4<monthly_allotment[j]) and (rake_allocated[j][w]==0)):
                combwh = np.array(np.nonzero(comb_matrix[j]==1))
                sum_ra = np.sum(rake_allocated,axis=1)
                remaining_aloc=monthly_allotment-sum_ra
                c=np.array(np.nonzero(remaining_aloc[combwh[0]]>0))
                possible_wh=combwh[0,c]
                num = np.size(possible_wh)
                if(num>0):
                    minimum=possible_wh[0,0]
                    temp=1
                    while(temp<num):
                      if(count_matrix[possible_wh[0,temp]][w]<count_matrix[minimum][w]):
                        minimum = possible_wh[0,temp]
                      temp+=1
                    cw = minimum
                    sum6 = sum_ra[cw]
                    while(rake_allocated[cw][w]==0 and sum6<monthly_allotment[cw] and dv2<partition):
                        rake_allocated[j][w] = 1
                        count_matrix[j][w] += 1
                        rake_allocated[cw][w] = 1
                        count_matrix[cw][w] += 1
                        comb[j][cw][w] = 1
                        comb[cw][j][w] = 1
                        allocated += 2
                        sum4 += 1
                        sum6 += 1
                    if(dv2>=partition and k>k_shift and (sum4+1)<monthly_allotment[j] and (terminal_capacity[j]==2 and rake_allocated[j][w]==0)):
                            if(sum4<(monthly_allotment[j]-1)):
#                                rake_allocated[j][w] = 2
#                                count_matrix[j][w] += 2
                               count_matrix,rake_allocated,comb,allocated,sum4=full_rake_split(j,w,rake_allocated,count_matrix,comb_matrix,comb,total_weeks,total_warehouses,weekly_penalty,monthly_allotment,allocated,sum4)
#                               allocated += 2
#                               sum4 += 2
                elif(num == 0 and (sum4+1)<monthly_allotment[j]):
#                    rake_allocated[j][w] = 2
#                    count_matrix[j][w] += 2
                   count_matrix,rake_allocated,comb,allocated,sum4=full_rake_split(j,w,rake_allocated,count_matrix,comb_matrix,comb,total_weeks,total_warehouses,weekly_penalty,monthly_allotment,allocated,sum4)
#                   allocated += 2
#                   sum4 += 2
            
        if(k%k_reset == 0):
            rake_allocated = np.zeros((total_warehouses,total_weeks))
            count_matrix=initial_count_matrix.copy()
            comb = np.zeros((total_warehouses,total_warehouses,total_weeks))
            allocated,rake_allocated,random_index,count_matrix = surplus_allocation(rake_allocated,monthly_allotment,total_warehouses,random_array,random_index,count_matrix,key,weekly_penalty,comb_matrix,comb,total_weeks)
        if(k>k_terminate):
            break
        k += 1
    return rake_allocated,comb,allocated,random_index,count_matrix
	
def penalty(rake_allocated,rake_penalty_h,rake_penalty_f,weekly_penalty,initial_stock_level,demand,storage_capacity,rake_penalty_half_half):
    half_rakes = np.array(np.nonzero(rake_allocated==1))
    half_half_rakes = np.array(np.nonzero(rake_allocated==2))
    full_rakes = np.array(np.nonzero(rake_allocated==3))
    size1 = np.size(half_rakes)
    size2 = np.size(full_rakes)
    size3 = np.size(half_half_rakes)
    cost1 = size1*rake_penalty_h/2
    cost2 = size2*rake_penalty_f/2
    cost3 = size3*rake_penalty_half_half/2
    rake_penalty = cost1+cost2+cost3

    alloted_weeks = np.array(np.nonzero(rake_allocated>0))
    week_penalty = 0
    weeknum = np.size(alloted_weeks)/2
    m = 0
    while m < weeknum:
        swh = alloted_weeks[0,m]
        sw = alloted_weeks[1,m]
        week_penalty += weekly_penalty[swh][sw]
        m += 1

    capacity_utilization_penalty = np.sum(capacity_utilization_pf(initial_stock_level+demand,storage_capacity))
    total_penalty = rake_penalty + week_penalty
    total_penalty_with_capacity=total_penalty+capacity_utilization_penalty
    return total_penalty,total_penalty_with_capacity
	
import time

def simulate(iteration,total_warehouses,total_weeks,total_rakes,demand,initial_stock_level,storage_capacity,terminal_capacity,max_allottment,rake_penalty_h,rake_penalty_f,weekly_penalty,k_shift,k_reset,k_terminate,count_matrix,comb_matrix,key,rake_penalty_half_half):
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
        allocated,rake_allocated,random_index,count_matrix = surplus_allocation(rake_allocated,demand,total_warehouses,random_array,random_index,count_matrix,key,weekly_penalty,comb_matrix,comb,total_weeks)
        rake_allocated,comb,allocated,random_index,count_matrix = rake_allocation(allocated,total_rakes,rake_allocated,monthly_allotment,comb_matrix,comb,k_shift,k_reset,k_terminate,terminal_capacity,total_warehouses,total_weeks,random_array,random_index,count_matrix,initial_count_matrix,key,weekly_penalty)
        
        total_penalty,total_penalty_with_capacity = penalty(rake_allocated,rake_penalty_h,rake_penalty_f,weekly_penalty,initial_stock_level,monthly_allotment,storage_capacity,rake_penalty_half_half)
        if(first ==1 and allocated == total_rakes):
            final_weekly_distribution = rake_allocated
            final_count_matrix=count_matrix
            final_total_penalty = total_penalty
            final_penalty_with_capacity= total_penalty_with_capacity
            final_comb = comb
            final_monthly_allotment=monthly_allotment
            first += 1
        if(allocated == total_rakes and first != 1):
            if(total_penalty_with_capacity<final_penalty_with_capacity):
                final_weekly_distribution = rake_allocated
                final_total_penalty = total_penalty
                final_penalty_with_capacity= total_penalty_with_capacity
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
    rake_penalty_half_half=40;
    weekly_penalty_rice =    np.array([10,1,1,1,10,1,1,1,10,1,1,2,10,1,1,1,2,2,1,1,10,1,1,1,3,1,1,1,5,2,1,4,5,2,1,1,2,2,1,1,10,1,1,4,4,2,1,2,4,1,1,2,3,2,1,1])
    weekly_penalty_wh =    np.array([10,2,1,1,10,2,1,1,10,1,1,1,3,2,1,2,1,1,1,1,10,1,1,1,4,1,2,1,10,1,1,2,7,3,1,2,10,2,1,1,10,1,1,1,1,2,2,1,10,1,1,1,10,1,1,1])
    k_shift = 135
    k_reset = 575
    k_terminate = 7450
    comb_matrix =    np.array([0,1,1,1,0,0,0,0,0,0,0,0,0,0,1,0,1,1,1,1,0,0,0,0,0,0,0,0,1,1,0,1,1,1,0,1,0,1,0,0,0,0,1,1,1,0,1,1,0,0,0,0,0,0,0,0,0,1,1,1,0,1,1,1,1,1,1,0,0,0,0,1,1,1,1,0,1,1,1,1,1,1,0,0,0,0,0,0,1,1,0,1,1,1,1,0,0,0,0,0,1,0,1,1,1,0,1,1,1,1,0,0,0,0,0,0,1,1,1,1,0,1,1,1,1,1,0,0,1,0,1,1,1,1,1,0,1,1,1,1,0,0,0,0,1,1,1,1,1,1,0,1,1,1,0,0,0,0,0,1,0,1,1,1,1,0,1,1,0,0,0,0,0,0,0,0,1,1,1,1,0,1,0,0,0,0,0,0,0,0,1,1,1,1,1,0])
    iteration=100
    count_matrix=np.zeros((total_warehouses,4))
    print("Boiled rice")
    count_matrix,new_stock_b,penalty_b,weekly_distribution_b=simulate(iteration,total_warehouses,total_weeks,total_rakes_b,demand_b,initial_stock_level_rice,storage_capacity,terminal_capacity,max_allottment,rake_penalty_h,rake_penalty_f,weekly_penalty_rice,k_shift,k_reset,k_terminate,count_matrix,comb_matrix,1,rake_penalty_half_half)
    
    print("Raw rice")
    initial_stock_level_rice=initial_stock_level_rice+new_stock_b
    count_matrix,new_stock_r,penalty_r,weekly_distribution_r=simulate(iteration,total_warehouses,total_weeks,total_rakes_r,demand_r,initial_stock_level_rice,storage_capacity,terminal_capacity,max_allottment,rake_penalty_h,rake_penalty_f,weekly_penalty_rice,k_shift,k_reset,k_terminate,count_matrix,comb_matrix,2,rake_penalty_half_half)
    
    initial_stock_level_rice=initial_stock_level_rice+new_stock_r
    initial_stock_level_wh= initial_stock_level_wh+initial_stock_level_rice
    print("Wheat") 
    count_matrix,new_stock_w,penalty_w,weekly_distribution_w=simulate(iteration,total_warehouses,total_weeks,total_rakes_w,demand_w,initial_stock_level_wh,storage_capacity,terminal_capacity,max_allottment,rake_penalty_h,rake_penalty_f,weekly_penalty_wh,k_shift,k_reset,k_terminate,count_matrix,comb_matrix,3,rake_penalty_half_half)
    #########finding cumulative values
    r_sum=find_sum_col(weekly_distribution_r)
    b_sum=find_sum_col(weekly_distribution_b)
    w_sum=find_sum_col(weekly_distribution_w)
    final_week_wise_sum=np.concatenate((w_sum,r_sum,b_sum),axis=0)
#    print(final_week_wise_sum)
    total_weekly_distribution=find_row_sum(weekly_distribution_b,weekly_distribution_r,weekly_distribution_w)
#    print(total_weekly_distribution_sum)
    print(count_matrix)
    
    
    penalty_sum=penalty_r+penalty_b+penalty_w
    total_new_stock=new_stock_b+new_stock_r+new_stock_w
    capacity_utilization_penalty = np.sum(capacity_utilization_pf(total_initial_stock_level+total_new_stock,storage_capacity))
    penalty_sum=penalty_sum+capacity_utilization_penalty 
    print(penalty_sum)
    end = time.time()
    print(end-start)
main()    