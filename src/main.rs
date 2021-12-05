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

fn main() {
    if false {
        day1(Part::One);
        day1(Part::Two);
        day2(Part::One);
    }
    day2(Part::Two);
}
