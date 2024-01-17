import numpy as np

class Reaction:
    '''
    A reaction is defined by [reactants], [products] and a [reaction constant].
    Both [reactants] and [products] are lists of strings, each element
    appearing the number of times it appears in the reaction.

    Example:

    reaction A + 3B -> C + 2D, rc=4.
    r1 = Reaction( ['A'] + 3*['B'], ['C' ,'D'] , 4.)
    '''

    def __init__(self, reactant, product, rc) -> None:
        self.reactant = dict((rea, reactant.count(rea)) for rea in set(reactant))
        self.product = dict((pro, product.count(pro)) for pro in set(product))
        self.rc = rc

        self.reaction_type = self.classify_reaction_type()

    def __repr__(self) -> str:

        def prettify(dictionnary) -> list:

            if len(dictionnary) == 0:
                return ['(/)']

            p_array = []

            for elt in dictionnary:
                num = dictionnary[elt]
                if num == 1:
                    p_array.append(elt)
                else:
                    p_array.append(str(num)+elt)

            return p_array

        return ' + '.join(prettify(self.reactant)) + ' -> '  + ' + '.join(prettify(self.product)) + f'  (rc={self.rc:.5f}) ({self.reaction_type})'

    def classify_reaction_type(self):

        reaction_type_dict = {0:'first order', 1:'second order', 2:'dimerization', 3:'production', 4:'other'}
        reactant = self.reactant

        # case PRODUCTION
        if len(reactant) == 0:
            return reaction_type_dict[3]

        # case FIRST ORDER
        if len(reactant) == 1 and list(reactant.values())[0] == 1:
            return reaction_type_dict[0]

        # case DIMERIZATION
        if len(reactant) == 1 and list(reactant.values())[0] == 2:
            return reaction_type_dict[2]

        # case SECOND ORDER 
        if len(reactant) == 2:
            return reaction_type_dict[1]

        # case OTHER
        return reaction_type_dict[4]

    def copy(self):
        react = [x for key,val in self.reactant.items() for x in val*[key]]
        prod = [x for key,val in self.product.items() for x in val*[key]]
        return Reaction(react, prod, self.rc)

class ReactionSet:
    '''
    Contains the information about the set of reactions.
    Can generate the stochioimetric matrix as well as the propensity function.
    The empty set element '(/)' is ignored when generating the list of elements as well as the stochioimetric matrix.
    '''

    def __init__(self, *args):

        self.reactions = []

        for arg in args:
            if not isinstance(arg, Reaction):
                raise Exception('A passed argument was not a Reaction object.')
            self.reactions.append(arg.copy())

        self.dist_elts()
        self.propensity_vector()


    def __repr__(self) -> str:

        def prettify_str(dictionnary) -> str:

            if len(dictionnary) == 0:
                return '(/)'

            p_array = []
            for elt in dictionnary:
                num = dictionnary[elt]
                if num == 1:
                    p_array.append(elt)
                else:
                    p_array.append(str(num)+elt)
            return ' + '.join(p_array)

        strs = [[], [], []]
        for reac in self.reactions:
            strs[0].append(prettify_str(reac.reactant))
            strs[1].append(prettify_str(reac.product))
            strs[2].append(reac.rc)

        max_lens = [0,0]
        max_lens[0] = max([len(str1) for str1 in strs[0]])
        max_lens[1] = max([len(str2) for str2 in strs[1]])

        s = 'Reactions: \n\n'
        for i in range(len(strs[0])):
            s = s + ('('+str(i+1)+')\t{0:<'+str(max_lens[0])+'} -> {1:<'+str(max_lens[1])+'}   (rc={2:<.5g})').format(strs[0][i], strs[1][i], strs[2][i]) + '\n'

        return s.rstrip('\n')

    def append(self, reac) -> None:
        self.reactions.append(reac.copy())

        self.dist_elts()
        self.propensity_vector()


    def removeReaction(self, indx) -> None:
        if indx < 1 or indx > len(self.reactions) or not isinstance(indx, int):
            raise Exception('Invalid index.')

        self.reactions.pop(indx-1)
        self.dist_elts()
        self.propensity_vector()


    def dist_elts(self) -> None:
        elts = set()
        for reac in self.reactions:
            elts = elts.union(set(reac.reactant.keys())).union(set(reac.product.keys()))

        elts = list(elts)
        elts.sort()

        self.elts = elts


    def stochioimetric_matrix(self, order=None):

        def stochioimetric_vector(rea, elts, order):

            if order is None:
                order = elts

            v = np.zeros((len(order),))
            keys_reactant = rea.reactant.keys()
            keys_product = rea.product.keys()

            for i, elt in enumerate(order):

                vp,vr = 0.,0.

                if elt in keys_reactant:
                    vr = rea.reactant[elt]
                if elt in keys_product:
                    vp = rea.product[elt]

                v[i] = vp - vr

            return v

        M = len(self.elts)
        N = len(self.reactions)

        SM = np.zeros((M,N))
        for n in range(N):
            SM[:,n] = stochioimetric_vector(self.reactions[n], self.elts, order=order)

        return SM

    def propensity_vector(self) -> None:
        '''
        Utility function for computing the propensity function values of each element.
        Sets the prop_vec list which contains information about the reaction rate, and the indexes of the
        reactants with respect to self.order_elts()
        '''
        a = []
        basic, dim, prod = [],[], []
        elt_loc = {val:i for i, val in enumerate(self.order_elts())}

        for i,reac in enumerate(self.reactions):
            elt = (reac.rc, [elt_loc[x] for x in reac.reactant.keys()])
            a.append(elt)

            if reac.reaction_type == 'dimerization':
                dim.append(i)
            elif reac.reaction_type == 'production':
                prod.append(i)
            else:
                basic.append(i)


        self.prop_vec = a
        self.reaction_types = {'dim':dim, 'basic':basic, 'prod':prod}

    def propensity_f(self, X):
        '''
        Computing the propensity function values given X.
        Currently works for first-order and second-order reactions, dimerisations, production reactions.
        For speed, makes use of information stored in [self.prop_vec] so this has to have been defined before.
        '''
        N = len(self.reactions)
        a = np.zeros((N,))

        for i in self.reaction_types['basic']:
            rc, indx = self.prop_vec[i]
            a[i] = rc * np.product(X[indx])

        for k in self.reaction_types['dim']:
            rc, indx = self.prop_vec[k]
            a[k] = rc * X[indx] * (X[indx] -1)

        for j in self.reaction_types['prod']:
            rc, indx = self.prop_vec[j]
            a[j] = rc

        return a

    def N(self, X):
        N = len(self.reactions)
        Nj = np.zeros(())

        for i in self.reaction_types['basic']:
            rc, indx = self.prop_vec[i]
            Nj[i] = np.amin(X[indx])

        for k in self.reaction_types['dim']:
            rc, indx = self.prop_vec[k]
            Nj[k] = np.floor(X[indx]/2)

        for j in self.reaction_types['prod']:
            rc, indx = self.prop_vec[j]
            Nj[j] = None

        return Nj


    def order_elts(self):
        return self.elts

    def order_reacs(self):
        return self.reactions


if __name__ == '__main__':

    nA = 6.023e23
    vol = 1e-15

    S1,S2,S3, S4 = ['S1'], ['S2'], ['S3'], ['S4']
    c1, c2, c3 = 1e6/(nA*vol), 1e-4, 0.1
    #r1 = Reaction(S1+S2, S3, c1)
    #r2 = Reaction(S3, S1+S2, c2)
    #r3 = Reaction(S3, S4+S2, c3)

    r1 = Reaction(S1, [], c1)
    r2 = Reaction([], S1, c2)

    X = np.zeros((1,))
    X[0] = round(5e-7*nA*vol)
    #X[1] = round(2e-7*nA*vol)

    rs = ReactionSet(r1, r2)
    print(rs)
