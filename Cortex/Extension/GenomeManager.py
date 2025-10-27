__all__ = ['GenomeManager']


from numpy import random
from abc import ABC, abstractmethod
from copy import deepcopy as COPY


class GenomeManager(ABC):
    def op_select(self, dst_genome_id = None, exclude_self = True):
        p_field = list(self.get_all_id())
        if dst_genome_id is None:
            dst_genome_id = self.new_genome()
            exclude_self = False
        if exclude_self: p_field.remove(dst_genome_id)
        p = random.choice(p_field)
        genome = self.get_genome(p)
        self.set_genome(dst_genome_id, genome)
        return p

    def op_crossover(self, dst_genome_id = None, cross_rate = 1, cross_point_num = 1, exclude_self = True):
        p_field = list(self.get_all_id())
        if dst_genome_id is None:
            dst_genome_id = self.new_genome()
            exclude_self = False
        if exclude_self: p_field.remove(dst_genome_id)
        if random.uniform() >= cross_rate: return None, None, None
        p0, p1 = random.choice(p_field, 2)
        genome0 = self.get_genome(p0)
        genome1 = self.get_genome(p1)
        len0 = len(genome0)
        len1 = len(genome1)
        cross_points = list(random.randint(0, min(len0, len1) + 1, cross_point_num))
        cross_points.sort()
        cross_points_0 = cross_points
        if cross_point_num % 2: cross_points_0 = [0] + cross_points
        replace = zip(cross_points_0[::2], cross_points_0[1::2])
        genome = genome0.copy()
        replace_len = 0
        for s0, s1 in replace:
            genome[s0 : s1] = genome1[s0 : s1]
            replace_len += s1 - s0
        self.set_genome(dst_genome_id, genome)
        if replace_len == 0: p1 = p0
        if replace_len == len0: p0 = p1
        return (p0, p1), ((len0 - replace_len) / len0, replace_len / len0), tuple(cross_points)

    def op_mutation(self, dst_genome_id, mut_rate):
        genome = self.get_genome(dst_genome_id)
        genome_len = len(genome)
        mut_num = random.poisson(genome_len * mut_rate)
        if mut_num == 0: return tuple()
        mut_sites = random.randint(0, genome_len, mut_num)
        for s in mut_sites: genome[s] = self.mutate_value(genome[s])
        self.set_genome(dst_genome_id, genome)
        return tuple(mut_sites)

    def genome_num(self): return len(self.get_all_id())

    def new_genome(self): raise NotImplementedError
    def remove_genome(self, genome_id): raise NotImplementedError

    @abstractmethod
    def get_genome(self, genome_id): raise NotImplementedError
    @abstractmethod
    def get_all_id(self): raise NotImplementedError
    @abstractmethod
    def set_genome(self, genome_id, genome): raise NotImplementedError
    @abstractmethod
    def mutate_value(self, v0): raise NotImplementedError
