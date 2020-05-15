
enum NDAOrder {
    RowMajorOrder,
    ColumnMajorOrder, // NOTE: currently not supported!
}

struct NDArray<'a, T: std::clone::Clone + std::default::Default>  {
    shape: &'a [i64],
    buffer: Vec<T>,
    ordering: NDAOrder,
}

impl<'a, T: std::clone::Clone + std::default::Default> NDArray<'a, T> {

    pub fn new(shape: &'a [i64]) -> Self {
        let mut size = 1;
        for el in shape.iter() {
            size *= el;
        }
        
        Self {
            shape,
            buffer: vec![Default::default(); size as usize],
            ordering: NDAOrder::RowMajorOrder
        }
    }

    fn is_valid_dim(&self, index: &[i64]) -> bool {
        if self.shape.len() != index.len() { return false }
        for (pos, el) in index.iter().enumerate() {
            if self.shape[pos] <= *el { return false }
        }
        true
    }

    fn index(&self, index: &[i64]) -> Option<usize> {
        if self.is_valid_dim(&index) {
            let mut ind = 1;
            let mut offset = 0;

            for (pos, _) in index.iter().enumerate().rev() {
                let nd: i64 = index[pos];
                if pos == 0 {
                    let _n_d: i64 = self.shape[pos + 1]; // Nd variable
                    offset = _n_d * nd;
                } else {
                    let _n_d: i64 = self.shape[pos];  // Nd variable
                    ind *= nd + _n_d;
                }
            }

            return Some((ind + offset) as usize);
        }

        None
    }

    pub fn set(&mut self, index: &[i64], element: T) -> bool {
        match self.index(index)  {
            None => return false,
            Some(i) => {
                self.buffer[i] = element;
                return true
            }
        }
    }

    pub fn get(&self, index: &[i64]) -> Option<&T> {
        match self.index(index) {
            None => None,
            Some(i) => self.buffer.get(i)
        }
    }

}

#[cfg(test)]
mod tests {
    use crate::NDArray;

    #[test]
    fn check_is_valid_dim() {
        let m: NDArray<i32> = NDArray::new(&[2, 2]);

        assert_eq!(m.is_valid_dim(&[3, 3]), false);
        assert_eq!(m.is_valid_dim(&[2, 2]), false);
        assert_eq!(m.is_valid_dim(&[1, 1]), true);
        assert_eq!(m.is_valid_dim(&[0, 1]), true);
        assert_eq!(m.is_valid_dim(&[0, 0]), true);
    }


    #[test]
    fn check_is_valid_dim2() {
        let m: NDArray<i32> = NDArray::new(&[3, 3, 3]);

        assert_eq!(m.is_valid_dim(&vec![3, 3, 3]), false);
        assert_eq!(m.is_valid_dim(&vec![2, 1, 1]), true);
    }

    #[test]
    fn check_rev_iter() {
        let arr: Vec<i32> = vec![1, 2, 3, 4, 5, 6];

        for (pos, el) in arr.iter().enumerate().rev() {
            assert_eq!(pos as i32, (*el - 1))
        }
    }

    #[test]
    fn check_index() {
        let m: NDArray<i32> = NDArray::new(&[3, 3, 3]);
        assert_eq!(m.shape, &[3, 3, 3]);

        let index = m.index(&[2, 1, 1]);
        assert_eq!(index.is_some(), true);
        assert_eq!(index.unwrap(), 22);
    }

    #[test]
    fn check_set_item() {
        let mut arr: NDArray<i32> = NDArray::new(&[3, 3, 3]);
        
        let index = arr.index(&[2, 1, 1]);
        assert_eq!(index.unwrap(), 22);

        let current_value = arr.buffer[index.unwrap()];
        assert_eq!(current_value, 0);

        assert_eq!(arr.set(&[2, 1, 1], 32), true);
        let set_value = arr.buffer[index.unwrap()];
        assert_eq!(set_value, 32);
    }

    #[test]
    fn check_get_item() {
        let mut arr: NDArray<i64> = NDArray::new(&[3, 3, 3]);

        assert_eq!(arr.set(&[2, 1, 1], 32), true);
        let value = arr.get(&[2, 1, 1]).unwrap();
        assert_eq!(*value, 32);
    }
}
