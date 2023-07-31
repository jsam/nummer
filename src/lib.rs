use std::clone::Clone;
use std::default::Default;

pub enum Order {
    RowMajor,
    ColumnMajor,
}

pub struct NDArray<T>
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

    pub fn seq_index(&self) -> Vec<Vec<i64>> {
        match self.ordering {
            Order::RowMajor => {
                let num_dims = self.shape.len();
                let total_elements = self.row_major_strides[0] * self.shape[0];

                let mut indexes = Vec::with_capacity(total_elements as usize);
                let mut current_index = vec![0; num_dims];

                for _ in 0..total_elements {
                    indexes.push(current_index.clone());

                    // Increment the innermost index
                    current_index[num_dims - 1] += 1;

                    // Propagate the carry to higher dimensions
                    for dim in (1..num_dims).rev() {
                        if current_index[dim] == self.shape[dim] {
                            current_index[dim] = 0;
                            current_index[dim - 1] += 1;
                        }
                    }
                }

                indexes
            }
            Order::ColumnMajor => {
                let num_dims = self.shape.len();
                let total_elements = self.row_major_strides[0] * self.shape[0];

                let mut indexes = Vec::with_capacity(total_elements as usize);
                let mut current_index = vec![0; num_dims];

                for _ in 0..total_elements {
                    indexes.push(current_index.clone());

                    // Increment the outermost index
                    current_index[0] += 1;

                    // Propagate the carry to higher dimensions
                    for dim in 0..num_dims - 1 {
                        if current_index[dim] == self.shape[dim] {
                            current_index[dim] = 0;
                            current_index[dim + 1] += 1;
                        }
                    }
                }

                indexes
            }
        }
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

    pub fn get_mut(&mut self, index: &[i64]) -> Option<&mut T> {
        match self.index(index) {
            None => None,
            Some(i) => self.buffer.get_mut(i),
        }
    }

    pub fn iter(&self) -> IndexSequenceIter<T> {
        IndexSequenceIter {
            nd_array: self,
            indexes: self.seq_index(),
            index_index: 0,
        }
    }
}

pub struct IndexSequenceIter<'a, T>
where
    T: Clone + Default,
{
    nd_array: &'a NDArray<T>,
    indexes: Vec<Vec<i64>>,
    index_index: usize,
}

impl<'a, T> Iterator for IndexSequenceIter<'a, T>
where
    T: Clone + Default,
{
    type Item = (Vec<i64>, &'a T);

    fn next(&mut self) -> Option<Self::Item> {
        if self.index_index >= self.indexes.len() {
            return None;
        }

        let index = &self.indexes[self.index_index];
        self.index_index += 1;

        let value = self.nd_array.get(index)?;
        Some((index.to_vec(), value))
    }
}

#[cfg(test)]
mod tests {
    use crate::{NDArray, Order};

    #[test]
    fn check_is_valid_dim() {
        {
            let m: NDArray<i32> = NDArray::new(&[2, 2]);

            assert_eq!(m.is_valid_dim(&[3, 3]), false);
            assert_eq!(m.is_valid_dim(&[2, 2]), false);
            assert_eq!(m.is_valid_dim(&[1, 1]), true);
            assert_eq!(m.is_valid_dim(&[0, 1]), true);
            assert_eq!(m.is_valid_dim(&[0, 0]), true);
        }

        {
            let m: NDArray<i32> = NDArray::new(&[3, 3, 3]);

            assert_eq!(m.is_valid_dim(&vec![3, 3, 3]), false);
            assert_eq!(m.is_valid_dim(&vec![2, 1, 1]), true);
        }
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

    #[test]
    fn check_strides_2d() {
        {
            let arr = NDArray::<i32>::new(&[2, 3]);
            assert_eq!(arr.row_major_strides, vec![3, 1]);
            assert_eq!(arr.col_major_strides, vec![1, 2]);
        }

        {
            let arr = NDArray::<i32>::new(&[3, 2]);
            assert_eq!(arr.row_major_strides, vec![2, 1]);
            assert_eq!(arr.col_major_strides, vec![1, 3]);
        }

        {
            let arr = NDArray::<i32>::new(&[3, 4]);
            assert_eq!(arr.row_major_strides, vec![4, 1]);
            assert_eq!(arr.col_major_strides, vec![1, 3]);
        }

        {
            let arr = NDArray::<i32>::new(&[5, 2]);
            assert_eq!(arr.row_major_strides, vec![2, 1]);
            assert_eq!(arr.col_major_strides, vec![1, 5]);
        }

        {
            let arr = NDArray::<i32>::new(&[5, 5]);
            assert_eq!(arr.row_major_strides, vec![5, 1]);
            assert_eq!(arr.col_major_strides, vec![1, 5]);
        }
    }

    #[test]
    fn check_strides_3d() {
        {
            let arr = NDArray::<i32>::new(&[2, 3, 4]);
            assert_eq!(arr.row_major_strides, vec![12, 4, 1]);
            assert_eq!(arr.col_major_strides, vec![1, 2, 6]);
        }

        {
            let arr = NDArray::<i32>::new(&[3, 2, 4]);
            assert_eq!(arr.row_major_strides, vec![8, 4, 1]);
            assert_eq!(arr.col_major_strides, vec![1, 3, 6]);
        }

        {
            let arr = NDArray::<i32>::new(&[3, 4, 2]);
            assert_eq!(arr.row_major_strides, vec![8, 2, 1]);
            assert_eq!(arr.col_major_strides, vec![1, 3, 12]);
        }

        {
            let arr = NDArray::<i32>::new(&[4, 3, 2]);
            assert_eq!(arr.row_major_strides, vec![6, 2, 1]);
            assert_eq!(arr.col_major_strides, vec![1, 4, 12]);
        }

        {
            let arr = NDArray::<i32>::new(&[4, 2, 3]);
            assert_eq!(arr.row_major_strides, vec![6, 3, 1]);
            assert_eq!(arr.col_major_strides, vec![1, 4, 8]);
        }

        {
            let arr = NDArray::<i32>::new(&[5, 5, 5]);
            assert_eq!(arr.row_major_strides, vec![25, 5, 1]);
            assert_eq!(arr.col_major_strides, vec![1, 5, 25]);
        }
    }

    #[test]
    fn check_strides_4d() {
        {
            let arr = NDArray::<i32>::new(&[2, 3, 4, 5]);
            assert_eq!(arr.row_major_strides, vec![60, 20, 5, 1]);
            assert_eq!(arr.col_major_strides, vec![1, 2, 6, 24]);
        }

        {
            let arr = NDArray::<i32>::new(&[3, 2, 3, 2]);
            assert_eq!(arr.row_major_strides, vec![12, 6, 2, 1]);
            assert_eq!(arr.col_major_strides, vec![1, 3, 6, 18]);
        }

        {
            {
                let arr = NDArray::<i32>::new(&[4, 4, 4, 4]);
                assert_eq!(arr.row_major_strides, vec![64, 16, 4, 1]);
                assert_eq!(arr.col_major_strides, vec![1, 4, 16, 64]);
            }
        }
    }

    #[test]
    fn check_seq_indexes_row_major() {
        {
            let arr = NDArray::<i32>::new(&[2, 3]);
            let seq = arr.seq_index();
            assert_eq!(seq.len(), 6);
            assert_eq!(
                seq,
                vec![
                    vec![0, 0],
                    vec![0, 1],
                    vec![0, 2],
                    vec![1, 0],
                    vec![1, 1],
                    vec![1, 2],
                ]
            );
        }

        {
            let arr = NDArray::<i32>::new(&[3, 2, 4]);
            let seq = arr.seq_index();
            assert_eq!(seq.len(), 24);
            assert_eq!(
                seq,
                vec![
                    vec![0, 0, 0],
                    vec![0, 0, 1],
                    vec![0, 0, 2],
                    vec![0, 0, 3],
                    vec![0, 1, 0],
                    vec![0, 1, 1],
                    vec![0, 1, 2],
                    vec![0, 1, 3],
                    vec![1, 0, 0],
                    vec![1, 0, 1],
                    vec![1, 0, 2],
                    vec![1, 0, 3],
                    vec![1, 1, 0],
                    vec![1, 1, 1],
                    vec![1, 1, 2],
                    vec![1, 1, 3],
                    vec![2, 0, 0],
                    vec![2, 0, 1],
                    vec![2, 0, 2],
                    vec![2, 0, 3],
                    vec![2, 1, 0],
                    vec![2, 1, 1],
                    vec![2, 1, 2],
                    vec![2, 1, 3],
                ]
            );
        }

        {
            let arr = NDArray::<i32>::new(&[2, 2, 2, 2]);
            let seq = arr.seq_index();
            assert_eq!(seq.len(), 16);
            assert_eq!(
                seq,
                vec![
                    vec![0, 0, 0, 0],
                    vec![0, 0, 0, 1],
                    vec![0, 0, 1, 0],
                    vec![0, 0, 1, 1],
                    vec![0, 1, 0, 0],
                    vec![0, 1, 0, 1],
                    vec![0, 1, 1, 0],
                    vec![0, 1, 1, 1],
                    vec![1, 0, 0, 0],
                    vec![1, 0, 0, 1],
                    vec![1, 0, 1, 0],
                    vec![1, 0, 1, 1],
                    vec![1, 1, 0, 0],
                    vec![1, 1, 0, 1],
                    vec![1, 1, 1, 0],
                    vec![1, 1, 1, 1],
                ]
            );
        }
    }

    #[test]
    fn check_seq_indexes_column_major() {
        {
            let mut arr = NDArray::<i32>::new(&[2, 3]);
            arr.ordering = Order::ColumnMajor;
            let seq = arr.seq_index();
            assert_eq!(seq.len(), 6);
            assert_eq!(
                seq,
                vec![
                    vec![0, 0],
                    vec![1, 0],
                    vec![0, 1],
                    vec![1, 1],
                    vec![0, 2],
                    vec![1, 2],
                ]
            );
        }

        {
            let mut arr = NDArray::<i32>::new(&[2, 3, 4]);
            arr.ordering = Order::ColumnMajor;
            let seq = arr.seq_index();
            assert_eq!(seq.len(), 24);
            assert_eq!(
                seq,
                vec![
                    vec![0, 0, 0],
                    vec![1, 0, 0],
                    vec![0, 1, 0],
                    vec![1, 1, 0],
                    vec![0, 2, 0],
                    vec![1, 2, 0],
                    vec![0, 0, 1],
                    vec![1, 0, 1],
                    vec![0, 1, 1],
                    vec![1, 1, 1],
                    vec![0, 2, 1],
                    vec![1, 2, 1],
                    vec![0, 0, 2],
                    vec![1, 0, 2],
                    vec![0, 1, 2],
                    vec![1, 1, 2],
                    vec![0, 2, 2],
                    vec![1, 2, 2],
                    vec![0, 0, 3],
                    vec![1, 0, 3],
                    vec![0, 1, 3],
                    vec![1, 1, 3],
                    vec![0, 2, 3],
                    vec![1, 2, 3],
                ]
            );
        }
    }

    #[test]
    fn check_example() {
        let array = NDArray::<i64>::new(&[5, 4]);

        array.iter().for_each(|(index, value)| {
            println!("{index:?} -> {value}");
            assert_eq!(*value, 0);
        });

        assert_eq!(
            array.iter().map(|x| *x.1).collect::<Vec<_>>(),
            vec![0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        )
    }
}
