from django import forms

class FirstForm(forms.Form):
    demand = forms.CharField(widget=forms.TextInput(attrs={'class': 'form-control','placeholder':'Enter Demands for Warehouses as [w1,w2,...,wn] for each Warehouse'}),initial='3,3,2,2,1,1,3,3',label='Demand',max_length=100)
    isl = forms.CharField(widget=forms.TextInput(attrs={'class': 'form-control','placeholder':'Enter Initial Stock Level as [w1,w2,...,wn] for each Warehouse'}),initial='8,21,4,3,15,4,35,3',label='Initial Stock Level',max_length=100)
    weekp = forms.CharField(widget=forms.TextInput(attrs={'class': 'form-control','placeholder':'Enter the Weekly Penalty'}),initial='1,1,10,2,4,2,2,1,2,2,3,1,2,2,3,1,4,4,3,1,2,3,4,1,4,4,3,1,4,7,3,1',label='Weekly Penalty',max_length=100)
    tr = forms.IntegerField(widget=forms.TextInput(attrs={'class': 'form-control','placeholder':'Enter the Total Number of Rakes'}),label='Total Number of Rakes')
    iterations = forms.IntegerField(widget=forms.TextInput(attrs={'class': 'form-control','placeholder':'Enter the Number of Iterations'}),label='Number of Iterations')
class SecForm(forms.Form):    
    combm = forms.CharField(widget=forms.TextInput(attrs={'class': 'form-control','placeholder':'Enter The Combination matrix [(w1w1),(w1w2),......] as 1 - Yes, 0 - No'}),initial='0,1,0,0,0,1,0,1,1,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,1,0,0,0,0,1,0,1,0,0,0,0,1,0,0,0,0,0,1,1,0,0,0,0,0,1,0,1,1,0,0,0,0,1,1,0',label='Combination Matrix',max_length=200)
    n = forms.IntegerField(widget=forms.TextInput(attrs={'class': 'form-control','placeholder':'Enter the Number of weeks'}),initial=4,label='Number of weeks')
    m = forms.IntegerField(widget=forms.TextInput(attrs={'class': 'form-control','placeholder':'Enter the Number of Warehouses'}),initial=8,label='Number of Warehouses')
    sc = forms.CharField(widget=forms.TextInput(attrs={'class': 'form-control','placeholder':'Enter Storage Capacity as [w1,w2,...,wn] for each Warehouse'}),label='Storage Capacity',initial='13,25,6,5,16,5,40,6',max_length=100)
    st = forms.CharField(widget=forms.TextInput(attrs={'class': 'form-control','placeholder':'Enter Terminal Capacity as [t1,t2,...,tn] for each Terminal in the Warehouse'}),initial='2,2,2,2,2,2,2,2',label='Terminal Capacity',max_length=100)
    rph = forms.IntegerField(widget=forms.TextInput(attrs={'class': 'form-control','placeholder':'Penalty for a Half Rake'}),initial=20,label='Half Rake Penalty')
    rpf = forms.IntegerField(widget=forms.TextInput(attrs={'class': 'form-control','placeholder':'Penalty for a Full Rake'}),initial=50,label='Full Rake Penalty')
    ma = forms.CharField(widget=forms.TextInput(attrs={'class': 'form-control','placeholder':'Enter Maximum Allotcation as [w1,w2,...,wn] for each Warehouse'}),initial='8,8,8,8,8,8,8,8',label='Maximum Allocation',max_length=100)
    kshift = forms.IntegerField(widget=forms.TextInput(attrs={'class': 'form-control','placeholder':'Enter the value of K-Shift'}),initial=375,label='KShift')
    kterm = forms.IntegerField(widget=forms.TextInput(attrs={'class': 'form-control','placeholder':'Enter the value of K-Terminate'}),initial=7450,label='KTerminate')
    kreset = forms.IntegerField(widget=forms.TextInput(attrs={'class': 'form-control','placeholder':'Enter the value of K-Reset'}),initial=575,label='KReset')

    

    def clean1(self):
        cleaned_data = super(FirstForm, self).clean()
        demand = cleaned_data.get('demand')
        isl = cleaned_data.get('isl')
        tr = cleaned_data.get('tr')
        iterations = cleaned_data.get('iterations')
        weekp = cleaned_data.get('weekp')
        
    def clean2(self): 
        cleaned_data = super(SecForm, self).clean()
        combm = cleaned_data.get('combm')
        n = cleaned_data.get('n')
        m = cleaned_data.get('m')
        st = cleaned_data.get('sc')
        rph = cleaned_data.get('rph')
        ma = cleaned_data.get('ma')
        kshift = cleaned_data.get('kshift')
        kterm = cleaned_data.get('kterm')
        kreset = cleaned_data.get('kreset')
        rpf = cleaned_data.get('rpf')
#        if not demand and not isl and not tr:
#            raise forms.ValidationError('You have to write something!')