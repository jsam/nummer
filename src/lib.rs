use std::clone::Clone;
use std::default::Default;

pub enum Order {
    RowMajor,
    ColumnMajor,
}

struct NDArray<T>
where
    T: Clone + Default,
{
    shape: Vec<i64>,
    buffer: Vec<T>,

    row_major_strides: Vec<i64>,
    col_major_strides: Vec<i64>,
    pub ordering: Order,
}

impl<T> NDArray<T>
where
    T: Clone + Default,
{
    fn strides(shape: &[i64]) -> (Vec<i64>, Vec<i64>) {
        let ndim = shape.len();
        let mut row_major_strides = vec![0; ndim];
        let mut col_major_strides = vec![0; ndim];

        // Calculate strides for row-major order
        let mut stride = 1;
        for i in (0..ndim).rev() {
            row_major_strides[i] = stride;
            stride *= shape[i];
        }

        // Calculate strides for column-major order
        stride = 1;
        for i in 0..ndim {
            col_major_strides[i] = stride;
            stride *= shape[i];
        }

        (row_major_strides, col_major_strides)
    }

    pub fn new(shape: &[i64]) -> Self {
        let mut size = 1;
        for el in shape.iter() {
            size *= el;
        }
        let (row_major_strides, col_major_strides) = Self::strides(shape);

        Self {
            shape: shape.to_vec(),
            buffer: vec![Default::default(); size as usize],
            row_major_strides,
            col_major_strides,
            ordering: Order::RowMajor,
        }
    }

    fn is_valid_dim(&self, index: &[i64]) -> bool {
        if self.shape.len() != index.len() {
            return false;
        }
        for (pos, el) in index.iter().enumerate() {
            if self.shape[pos] <= *el {
                return false;
            }
        }
        true
    }

    fn index(&self, index: &[i64]) -> Option<usize> {
        if self.is_valid_dim(index) {
            let mut offset = 0;

            match self.ordering {
                Order::RowMajor => {
                    for (pos, &nd) in index.iter().enumerate().rev() {
                        offset += nd * self.row_major_strides[pos];
                    }
                }
                Order::ColumnMajor => {
                    for (pos, &nd) in index.iter().enumerate() {
                        offset += nd * self.col_major_strides[pos];
                    }
                }
            }

            Some(offset as usize)
        } else {
            None
        }
    }

    pub fn set(&mut self, index: &[i64], element: T) -> bool {
        match self.index(index) {
            None => return false,
            Some(i) => {
                self.buffer[i] = element;
                return true;
            }
        }
    }

    pub fn get(&self, index: &[i64]) -> Option<&T> {
        match self.index(index) {
            None => None,
            Some(i) => self.buffer.get(i),
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
