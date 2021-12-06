#![feature(drain_filter)]
use itertools::Itertools;

#[derive(PartialEq, Eq)]
enum Part {
    One,
    Two,
}

fn day1(part: Part) {
    let depths = include_str!("day1_input.txt");
    let depths: Vec<i32> = depths.lines().map(|line| line.parse().unwrap()).collect();

    match part {
        Part::One => {
            let n_larger = depths
                .windows(2)
                .filter(|slice| slice[1] > slice[0])
                .count();
            println!("{}", n_larger);
        }
        Part::Two => {
            let window_sums = depths
                .windows(3)
                .map(|window| -> i32 { window.iter().sum() });
            let n_larger = window_sums
                .tuple_windows()
                .filter(|(sum, next_sum)| next_sum > sum)
                .count();
            println!("{}", n_larger);
        }
    }
}

fn day2(part: Part) {
    let navigational_commands = include_str!("day2_input.txt");
    let mut horizontal_pos = 0;
    let mut depth = 0;
    let mut aim = 0;

    for command in navigational_commands.lines() {
        let (command, arg) = command.split_once(' ').unwrap();
        let arg: i64 = arg.parse().unwrap();

        match part {
            Part::One => match command {
                "forward" => horizontal_pos += arg,
                "down" => depth += arg,
                "up" => depth -= arg, // what happens if the depth turns out negative?
                _ => unreachable!(),
            },
            Part::Two => match command {
                "forward" => {
                    horizontal_pos += arg;
                    depth += aim * arg;
                }
                "down" => aim += arg,
                "up" => aim -= arg,
                _ => unreachable!(),
            },
        }
    }

    println!("{}", horizontal_pos * depth);
}

fn bit_iter(num: i64, n_len: usize) -> impl Iterator<Item = bool> {
    (0..n_len).map(move |pos| (1 << pos) & num != 0)
}

fn day3(part: Part) {
    let nums = include_str!("day3_input.txt");
    //let nums = include_str!("day3_test_input.txt");
    let n_width = nums.lines().next().unwrap().len();
    let nums = nums
        .lines()
        .map(|n| i64::from_str_radix(n, 2).unwrap())
        .collect::<Vec<_>>();

    let count_bit_occurences = |nums: &[i64]| {
        // 1 integer per bit position
        // add 1, if the bit is set for a number, subtract 1 if it's not set.
        // => >0 => more set than not set, -1 => opposite, 0 => equal
        // little endian
        let mut sum_for_bit = vec![0; n_width];

        for num in nums.iter() {
            for (bit_sum, bit) in sum_for_bit.iter_mut().zip(bit_iter(*num, n_width)) {
                *bit_sum += if bit { 1 } else { -1 };
            }
        }
        sum_for_bit
    };

    match part {
        Part::One => {
            let sum_for_bit = count_bit_occurences(&nums);
            let gamma = sum_for_bit.iter().rev().fold(0, |num, bit| {
                let most_common_bit = match bit.cmp(&0) {
                    std::cmp::Ordering::Less => 0,
                    std::cmp::Ordering::Equal => unreachable!(), // undefined state
                    std::cmp::Ordering::Greater => 1,
                };
                (num << 1) | most_common_bit
            });
            let epsilon = !gamma & ((1 << n_width) - 1);
            println!("{} * {} = {}", gamma, epsilon, gamma * epsilon);
        }
        Part::Two => {
            let find_rating = |filter_least_common: bool| {
                let mut candidates = nums.clone();

                let mut bit_pos = n_width - 1;

                while candidates.len() > 1 {
                    let sum_for_bit = count_bit_occurences(&candidates);
                    let most_common_bit = if ((sum_for_bit[bit_pos]) >= 0) ^ filter_least_common {
                        1
                    } else {
                        0
                    };
                    candidates.retain(|n| ((n >> bit_pos) & 1) == most_common_bit);
                    if bit_pos == 0 {
                        break;
                    }
                    bit_pos -= 1;
                }

                assert!(candidates.len() == 1);

                candidates[0]
            };
            let oxygen_rating = find_rating(false);
            let scrubber_rating = find_rating(true);
            let life_support_rating = oxygen_rating * scrubber_rating;
            println!(
                "{} * {} = {}",
                oxygen_rating, scrubber_rating, life_support_rating
            );
            println!(
                "binary: {:b} * {:b} = {:b}",
                oxygen_rating, scrubber_rating, life_support_rating
            );
        }
    }
}

fn parse_num(num: &str) -> i64 {
    num.parse().unwrap()
}

struct BingoBoard {
    nums: Vec<i64>,
    marked: [bool; 25],
}

impl BingoBoard {
    fn parse(board: &str) -> Self {
        Self {
            nums: board.split_whitespace().map(parse_num).collect(),
            marked: [false; 25],
        }
    }

    // mark field if the number is part of the board
    // and return the score of the board if it wins because of it
    fn mark_num(&mut self, num: i64) -> Option<i64> {
        if let Some((idx, _)) = self
            .nums
            .iter()
            .enumerate()
            .find(|(_, board_num)| num == **board_num)
        {
            self.marked[idx] = true;

            if self.is_bingo() {
                let sum_unmarked: i64 = self
                    .nums
                    .iter()
                    .zip(self.marked)
                    .filter(|(_, marked)| !*marked)
                    .map(|(num, _)| num)
                    .sum();
                return Some(num * sum_unmarked);
            }
        }
        None
    }

    fn is_bingo(&self) -> bool {
        (0..5).any(|row| (5 * row..5 * row + 5).all(|idx| self.marked[idx]))
            || (0..5).any(|col| (col..25).step_by(5).all(|idx| self.marked[idx]))
    }
}

fn day4(part: Part) {
    let input = include_str!("day4_input.txt");
    //let input = include_str!("day4_test_input.txt");
    let (all_nums, fields) = input.split_once("\n").unwrap();
    let all_nums = all_nums.split(',').map(parse_num).collect::<Vec<_>>();
    let mut boards = fields
        .split("\n\n")
        .map(BingoBoard::parse)
        .collect::<Vec<_>>();

    match part {
        Part::One => {
            'outer: for num in all_nums {
                for board in boards.iter_mut() {
                    if let Some(winning_score) = board.mark_num(num) {
                        println!("{}", winning_score);
                        break 'outer;
                    }
                }
            }
        }
        Part::Two => {
            // assuming all boards win eventually
            let mut nums = all_nums.iter();
            for &num in nums.by_ref() {
                // .retain() doesn't give us an &mut
                boards.drain_filter(|board| board.mark_num(num).is_some());
                if boards.len() == 1 {
                    break;
                }
            }
            let mut last_board = boards.into_iter().next().unwrap();
            let score = nums.find_map(|num| last_board.mark_num(*num)).unwrap();
            println!("{}", score);
        }
    }
}

fn main() {
    if false {
        day1(Part::One);
        day1(Part::Two);
        day2(Part::One);
        day2(Part::Two);
        day3(Part::One);
        day3(Part::Two);
        day4(Part::One);
    }
    day4(Part::Two);
}
